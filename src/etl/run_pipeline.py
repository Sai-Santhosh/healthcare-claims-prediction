"""
ETL Pipeline Runner
Command-line interface for running the data engineering pipeline.
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.etl.pipeline import ETLPipeline, ETLPipelineBuilder
from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer, DataCleaner
from src.etl.load import DataLoader, S3Loader
from src.data_quality.expectations import DataQualityChecker, ExpectationSuite
from src.monitoring.metrics import MetricsCollector
from src.monitoring.alerting import AlertManager
from src.catalog.lineage import LineageTracker, OperationType

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load pipeline configuration."""
    if config_path is None:
        config_path = project_root / "config" / "settings.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_expectation_suite() -> ExpectationSuite:
    """Create data quality expectation suite."""
    suite = ExpectationSuite("claims_validation")
    
    suite.expect_table_row_count_to_be_between(1000, 50000000)
    suite.expect_column_to_exist("claim_id_key")
    suite.expect_column_values_to_not_be_null("claim_id_key", mostly=0.99)
    suite.expect_column_values_to_be_positive("amt_paid", mostly=0.95)
    suite.expect_column_values_to_be_positive("amt_billed", mostly=0.95)
    suite.expect_column_values_to_be_between("age", min_value=0, max_value=120, mostly=0.99)
    
    return suite


def build_pipeline(config: dict, args: argparse.Namespace) -> ETLPipeline:
    """Build the ETL pipeline based on configuration."""
    
    # Create expectation suite
    suite = create_expectation_suite()
    
    # Build pipeline using builder pattern
    builder = ETLPipelineBuilder(
        name=args.name or "claims_etl_pipeline",
        config=config
    )
    
    # Add extract stage
    builder.add_extract(DataExtractor(
        name="extract",
        chunk_size=config.get('etl', {}).get('chunk_size', 100000),
        delimiter="|",
        retry_attempts=3
    ))
    
    # Add transform stage
    builder.add_transform(DataTransformer(
        name="transform",
        transformations=[
            "clean_missing",
            "standardize_columns",
            "encode_categoricals",
            "create_derived_features"
        ]
    ))
    
    # Add quality check stage
    builder.add_quality_check(DataQualityChecker(
        name="quality_check",
        suite=suite,
        fail_on_error=args.fail_on_quality_error,
        reports_dir=str(project_root / "reports" / "quality")
    ))
    
    # Add load stage
    if args.output_to_s3:
        builder.add_load(S3Loader(
            name="load_to_s3",
            bucket=config.get('aws', {}).get('s3', {}).get('data_lake_bucket'),
            prefix="silver/claims",
            layer="silver"
        ))
    else:
        builder.add_load(DataLoader(
            name="load_local",
            output_path=str(project_root / "data" / args.layer / "claims_processed.parquet"),
            output_format="parquet"
        ))
    
    return builder.build()


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the ETL pipeline."""
    
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config or 'default'}")
    
    # Initialize tracking components
    metrics = MetricsCollector()
    lineage = LineageTracker(str(project_root / "data" / "lineage"))
    alert_manager = AlertManager(config.get('aws', {}))
    
    # Build pipeline
    pipeline = build_pipeline(config, args)
    logger.info(f"Pipeline: {pipeline.name} (ID: {pipeline.pipeline_id})")
    logger.info(f"Stages: {[s.name for s in pipeline.stages]}")
    
    # Start lineage tracking
    lineage.start_job(pipeline.pipeline_id, pipeline.name)
    lineage.register_source(
        "raw_claims",
        "PUBLICUSE_CLAIM_MC_2016.txt",
        {"rows": 16982295, "size_gb": 3.73}
    )
    
    # Execute pipeline
    result = pipeline.run()
    
    # Track lineage
    lineage.track_transformation(
        source_ids=["raw_claims"],
        target_id=f"{args.layer}_claims",
        operation=OperationType.TRANSFORM,
        transformation_logic="ETL pipeline: clean, transform, validate"
    )
    lineage.register_target(
        f"{args.layer}_claims",
        f"claims_processed.parquet",
        {"layer": args.layer}
    )
    lineage.end_job(status=result.status.value)
    
    # Record metrics
    metrics.record_pipeline_completion(
        pipeline_name=pipeline.name,
        status=result.status.value,
        duration=result.total_duration_seconds,
        rows_processed=result.total_rows_processed
    )
    metrics.flush()
    
    # Send notifications
    if result.status.value == "success":
        logger.info("Sending success notification...")
        alert_manager.send_pipeline_success_alert(
            pipeline_name=pipeline.name,
            duration_seconds=result.total_duration_seconds,
            rows_processed=result.total_rows_processed
        )
    else:
        logger.error("Sending failure notification...")
        alert_manager.send_pipeline_failure_alert(
            pipeline_name=pipeline.name,
            stage_name=result.stages[-1].stage_name if result.stages else "unknown",
            error_message=result.error_message or "Unknown error"
        )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Status: {result.status.value.upper()}")
    logger.info(f"Duration: {result.total_duration_seconds:.2f} seconds")
    logger.info(f"Rows Processed: {result.total_rows_processed:,}")
    logger.info(f"Rows Failed: {result.total_rows_failed:,}")
    
    logger.info("\nStage Results:")
    for stage in result.stages:
        status_icon = "✓" if stage.status.value == "success" else "✗"
        logger.info(
            f"  {status_icon} {stage.stage_name}: {stage.status.value} "
            f"({stage.duration_seconds:.2f}s, {stage.rows_processed:,} rows)"
        )
    
    if result.error_message:
        logger.error(f"\nError: {result.error_message}")
    
    logger.info("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if result.status.value == "success" else 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Claims ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.etl.run_pipeline
  python -m src.etl.run_pipeline --name my_pipeline --layer silver
  python -m src.etl.run_pipeline --output-to-s3 --fail-on-quality-error
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/settings.yaml)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Pipeline name (default: claims_etl_pipeline)"
    )
    
    parser.add_argument(
        "--layer",
        type=str,
        choices=["bronze", "silver", "gold"],
        default="silver",
        help="Target data layer (default: silver)"
    )
    
    parser.add_argument(
        "--output-to-s3",
        action="store_true",
        help="Output to S3 instead of local storage"
    )
    
    parser.add_argument(
        "--fail-on-quality-error",
        action="store_true",
        help="Fail pipeline if data quality checks fail"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        config = load_config(args.config)
        pipeline = build_pipeline(config, args)
        print(f"Pipeline: {pipeline.name}")
        print(f"Stages: {[s.name for s in pipeline.stages]}")
        print("Dry run - no execution")
        return
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
