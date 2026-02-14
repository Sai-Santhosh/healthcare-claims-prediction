"""
ETL Pipeline Orchestrator
Production-grade pipeline orchestration with monitoring, alerting, and lineage tracking.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from ..utils.logger import get_logger
from ..monitoring.metrics import MetricsCollector
from ..monitoring.alerting import AlertManager

logger = get_logger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(Enum):
    """Individual stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    rows_processed: int = 0
    rows_failed: int = 0
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    stages: List[StageResult] = field(default_factory=list)
    total_rows_processed: int = 0
    total_rows_failed: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "total_rows_processed": self.total_rows_processed,
            "total_rows_failed": self.total_rows_failed,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "status": s.status.value,
                    "duration_seconds": s.duration_seconds,
                    "rows_processed": s.rows_processed,
                }
                for s in self.stages
            ],
        }


class PipelineStage:
    """
    Base class for pipeline stages.
    Implements retry logic, metrics, and error handling.
    """
    
    def __init__(
        self,
        name: str,
        retry_attempts: int = 3,
        retry_delay: int = 30
    ):
        self.name = name
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.metrics = MetricsCollector()
    
    def execute(self, context: Dict[str, Any]) -> StageResult:
        """
        Execute the stage with retry logic.
        
        Args:
            context: Pipeline context with data and configuration
            
        Returns:
            StageResult with execution details
        """
        result = StageResult(
            stage_name=self.name,
            status=StageStatus.RUNNING,
            start_time=datetime.now()
        )
        
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Stage '{self.name}' - Attempt {attempt + 1}/{self.retry_attempts}")
                
                # Execute stage logic
                output = self._run(context)
                
                # Update result
                result.status = StageStatus.SUCCESS
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.rows_processed = output.get("rows_processed", 0)
                result.output_path = output.get("output_path")
                result.metadata = output.get("metadata", {})
                
                # Record metrics
                self.metrics.record_stage_success(self.name, result.duration_seconds)
                
                logger.info(
                    f"Stage '{self.name}' completed successfully in {result.duration_seconds:.2f}s. "
                    f"Processed {result.rows_processed:,} rows."
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Stage '{self.name}' failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.retry_attempts - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    result.status = StageStatus.FAILED
                    result.end_time = datetime.now()
                    result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                    result.error_message = str(e)
                    
                    self.metrics.record_stage_failure(self.name, str(e))
                    
                    return result
        
        return result
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage execution logic. Override in subclasses.
        
        Args:
            context: Pipeline context
            
        Returns:
            Dictionary with execution outputs
        """
        raise NotImplementedError("Subclasses must implement _run method")


class ETLPipeline:
    """
    Main ETL Pipeline Orchestrator.
    Manages stage execution, monitoring, alerting, and lineage tracking.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        stages: Optional[List[PipelineStage]] = None
    ):
        self.name = name
        self.config = config
        self.stages: List[PipelineStage] = stages or []
        self.metrics = MetricsCollector()
        self.alert_manager = AlertManager(config.get("aws", {}))
        
        # Pipeline state
        self.pipeline_id = str(uuid.uuid4())
        self.context: Dict[str, Any] = {}
        
        logger.info(f"Initialized ETL Pipeline: {name} (ID: {self.pipeline_id})")
    
    def add_stage(self, stage: PipelineStage) -> "ETLPipeline":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        logger.info(f"Added stage: {stage.name}")
        return self
    
    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Args:
            initial_context: Initial data and configuration
            
        Returns:
            PipelineResult with complete execution details
        """
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.context = initial_context or {}
        self.context["config"] = self.config
        self.context["pipeline_id"] = self.pipeline_id
        
        logger.info("=" * 60)
        logger.info(f"Starting Pipeline: {self.name}")
        logger.info(f"Pipeline ID: {self.pipeline_id}")
        logger.info(f"Stages: {[s.name for s in self.stages]}")
        logger.info("=" * 60)
        
        try:
            for stage in self.stages:
                logger.info(f"\n{'─' * 40}")
                logger.info(f"Executing Stage: {stage.name}")
                logger.info(f"{'─' * 40}")
                
                stage_result = stage.execute(self.context)
                result.stages.append(stage_result)
                result.total_rows_processed += stage_result.rows_processed
                result.total_rows_failed += stage_result.rows_failed
                
                # Update context with stage output
                if stage_result.output_path:
                    self.context[f"{stage.name}_output"] = stage_result.output_path
                
                # Check for failure
                if stage_result.status == StageStatus.FAILED:
                    result.status = PipelineStatus.FAILED
                    result.error_message = f"Stage '{stage.name}' failed: {stage_result.error_message}"
                    
                    # Send alert
                    self.alert_manager.send_pipeline_failure_alert(
                        pipeline_name=self.name,
                        stage_name=stage.name,
                        error_message=stage_result.error_message
                    )
                    
                    break
            
            # Pipeline completed successfully
            if result.status == PipelineStatus.RUNNING:
                result.status = PipelineStatus.SUCCESS
                
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Pipeline failed with unexpected error: {e}")
            
            self.alert_manager.send_pipeline_failure_alert(
                pipeline_name=self.name,
                stage_name="UNKNOWN",
                error_message=str(e)
            )
        
        finally:
            result.end_time = datetime.now()
            result.total_duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Record pipeline metrics
            self.metrics.record_pipeline_completion(
                pipeline_name=self.name,
                status=result.status.value,
                duration=result.total_duration_seconds,
                rows_processed=result.total_rows_processed
            )
            
            # Log summary
            self._log_summary(result)
        
        return result
    
    def _log_summary(self, result: PipelineResult) -> None:
        """Log pipeline execution summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Pipeline: {result.pipeline_name}")
        logger.info(f"Status: {result.status.value.upper()}")
        logger.info(f"Duration: {result.total_duration_seconds:.2f} seconds")
        logger.info(f"Total Rows Processed: {result.total_rows_processed:,}")
        logger.info(f"Total Rows Failed: {result.total_rows_failed:,}")
        
        logger.info("\nStage Results:")
        for stage in result.stages:
            status_icon = "✓" if stage.status == StageStatus.SUCCESS else "✗"
            logger.info(
                f"  {status_icon} {stage.stage_name}: {stage.status.value} "
                f"({stage.duration_seconds:.2f}s, {stage.rows_processed:,} rows)"
            )
        
        if result.error_message:
            logger.error(f"\nError: {result.error_message}")
        
        logger.info("=" * 60)


class ETLPipelineBuilder:
    """
    Builder pattern for constructing ETL pipelines.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.pipeline = ETLPipeline(name, config)
    
    def add_extract(self, extractor: PipelineStage) -> "ETLPipelineBuilder":
        """Add extraction stage."""
        self.pipeline.add_stage(extractor)
        return self
    
    def add_transform(self, transformer: PipelineStage) -> "ETLPipelineBuilder":
        """Add transformation stage."""
        self.pipeline.add_stage(transformer)
        return self
    
    def add_load(self, loader: PipelineStage) -> "ETLPipelineBuilder":
        """Add loading stage."""
        self.pipeline.add_stage(loader)
        return self
    
    def add_quality_check(self, checker: PipelineStage) -> "ETLPipelineBuilder":
        """Add data quality check stage."""
        self.pipeline.add_stage(checker)
        return self
    
    def build(self) -> ETLPipeline:
        """Build and return the pipeline."""
        return self.pipeline
