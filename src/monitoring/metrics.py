"""
Metrics Collection Module
Collects and reports pipeline metrics to CloudWatch and local storage.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineMetric:
    """Individual pipeline metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": self.dimensions
        }


class MetricsCollector:
    """
    Collects pipeline metrics for monitoring and alerting.
    Supports local storage and CloudWatch integration.
    """
    
    def __init__(
        self,
        namespace: str = "ClaimsPipeline",
        metrics_dir: str = "logs/metrics"
    ):
        self.namespace = namespace
        self.metrics_dir = metrics_dir
        self.metrics: List[PipelineMetric] = []
        
        os.makedirs(metrics_dir, exist_ok=True)
    
    def record_counter(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a counter metric."""
        self._record_metric(name, value, "Count", dimensions)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge metric."""
        self._record_metric(name, value, "None", dimensions)
    
    def record_timing(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing metric in seconds."""
        self._record_metric(name, value, "Seconds", dimensions)
    
    def _record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        metric = PipelineMetric(
            metric_name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            dimensions=dimensions or {}
        )
        
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def record_stage_success(self, stage_name: str, duration: float) -> None:
        """Record successful stage execution."""
        self.record_counter(
            "StageSuccess",
            1,
            {"StageName": stage_name}
        )
        self.record_timing(
            "StageDuration",
            duration,
            {"StageName": stage_name}
        )
    
    def record_stage_failure(self, stage_name: str, error: str) -> None:
        """Record failed stage execution."""
        self.record_counter(
            "StageFailure",
            1,
            {"StageName": stage_name, "ErrorType": error[:50]}
        )
    
    def record_pipeline_completion(
        self,
        pipeline_name: str,
        status: str,
        duration: float,
        rows_processed: int
    ) -> None:
        """Record pipeline completion metrics."""
        dimensions = {"PipelineName": pipeline_name, "Status": status}
        
        self.record_timing("PipelineDuration", duration, dimensions)
        self.record_counter("RowsProcessed", rows_processed, dimensions)
        
        if status == "success":
            self.record_counter("PipelineSuccess", 1, dimensions)
        else:
            self.record_counter("PipelineFailure", 1, dimensions)
    
    def record_data_quality_score(
        self,
        table_name: str,
        score: float,
        checks_passed: int,
        checks_total: int
    ) -> None:
        """Record data quality metrics."""
        dimensions = {"TableName": table_name}
        
        self.record_gauge("DataQualityScore", score, dimensions)
        self.record_counter("QualityChecksPassed", checks_passed, dimensions)
        self.record_counter("QualityChecksTotal", checks_total, dimensions)
    
    def flush(self) -> None:
        """Flush metrics to storage."""
        if not self.metrics:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(
                [m.to_dict() for m in self.metrics],
                f,
                indent=2
            )
        
        logger.info(f"Flushed {len(self.metrics)} metrics to {filepath}")
        self.metrics.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {
            "total_metrics": len(self.metrics),
            "by_name": {}
        }
        
        for metric in self.metrics:
            if metric.metric_name not in summary["by_name"]:
                summary["by_name"][metric.metric_name] = []
            summary["by_name"][metric.metric_name].append(metric.value)
        
        return summary


class CloudWatchMetrics:
    """
    AWS CloudWatch metrics integration.
    Publishes pipeline metrics to CloudWatch for monitoring and alerting.
    """
    
    def __init__(
        self,
        namespace: str = "ClaimsPipeline",
        region: str = "us-east-1"
    ):
        self.namespace = namespace
        self.region = region
        self._client = None
    
    @property
    def client(self):
        """Lazy-load CloudWatch client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('cloudwatch', region_name=self.region)
            except ImportError:
                logger.warning("boto3 not available. CloudWatch metrics disabled.")
                self._client = None
        return self._client
    
    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Put a metric to CloudWatch.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit (Count, Seconds, Bytes, etc.)
            dimensions: Metric dimensions
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            return False
        
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v}
                    for k, v in dimensions.items()
                ]
            
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
            
            logger.debug(f"Published CloudWatch metric: {metric_name}={value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish CloudWatch metric: {e}")
            return False
    
    def put_pipeline_metrics(
        self,
        pipeline_name: str,
        stage_name: str,
        rows_processed: int,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Put standard pipeline metrics."""
        dimensions = {
            "PipelineName": pipeline_name,
            "StageName": stage_name
        }
        
        self.put_metric("RowsProcessed", rows_processed, "Count", dimensions)
        self.put_metric("StageDuration", duration_seconds, "Seconds", dimensions)
        self.put_metric(
            "StageSuccess" if success else "StageFailure",
            1,
            "Count",
            dimensions
        )
    
    def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "GreaterThanThreshold",
        period: int = 300,
        evaluation_periods: int = 1,
        sns_topic_arn: Optional[str] = None
    ) -> bool:
        """
        Create a CloudWatch alarm.
        
        Args:
            alarm_name: Name for the alarm
            metric_name: Metric to monitor
            threshold: Alarm threshold
            comparison: Comparison operator
            period: Evaluation period in seconds
            evaluation_periods: Number of periods to evaluate
            sns_topic_arn: SNS topic for notifications
            
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            alarm_config = {
                'AlarmName': alarm_name,
                'MetricName': metric_name,
                'Namespace': self.namespace,
                'Statistic': 'Sum',
                'Period': period,
                'EvaluationPeriods': evaluation_periods,
                'Threshold': threshold,
                'ComparisonOperator': comparison
            }
            
            if sns_topic_arn:
                alarm_config['AlarmActions'] = [sns_topic_arn]
            
            self.client.put_metric_alarm(**alarm_config)
            
            logger.info(f"Created CloudWatch alarm: {alarm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alarm: {e}")
            return False
    
    def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get metric statistics from CloudWatch."""
        if self.client is None:
            return {}
        
        try:
            response = self.client.get_metric_statistics(
                Namespace=self.namespace,
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=statistics or ['Sum', 'Average', 'Maximum', 'Minimum']
            )
            
            return response.get('Datapoints', [])
            
        except Exception as e:
            logger.error(f"Failed to get metric statistics: {e}")
            return {}
