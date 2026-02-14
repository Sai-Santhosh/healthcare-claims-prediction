"""
Alerting Module
Handles pipeline alerts via AWS SNS and other channels.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message structure."""
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }
    
    def to_sns_message(self) -> str:
        """Format alert for SNS."""
        return f"""
========================================
{self.severity.value.upper()}: {self.title}
========================================

Source: {self.source}
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{self.message}

----------------------------------------
Metadata:
{json.dumps(self.metadata or {}, indent=2)}
========================================
"""


class AlertManager:
    """
    Central alert manager for pipeline notifications.
    Coordinates multiple alerting channels.
    """
    
    def __init__(
        self,
        aws_config: Optional[Dict[str, Any]] = None,
        alerts_dir: str = "logs/alerts"
    ):
        self.aws_config = aws_config or {}
        self.alerts_dir = alerts_dir
        self.alerters: List['BaseAlerter'] = []
        
        os.makedirs(alerts_dir, exist_ok=True)
        
        # Initialize SNS alerter if configured
        sns_config = self.aws_config.get('sns', {})
        if sns_config.get('topic_name'):
            self.alerters.append(
                SNSAlerter(
                    topic_name=sns_config['topic_name'],
                    region=self.aws_config.get('region', 'us-east-1')
                )
            )
        
        # Always add file alerter for local logging
        self.alerters.append(FileAlerter(alerts_dir))
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through all configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if at least one channel succeeded
        """
        success = False
        
        for alerter in self.alerters:
            try:
                if alerter.send(alert):
                    success = True
            except Exception as e:
                logger.error(f"Alerter {alerter.__class__.__name__} failed: {e}")
        
        return success
    
    def send_pipeline_failure_alert(
        self,
        pipeline_name: str,
        stage_name: str,
        error_message: str
    ) -> bool:
        """Send pipeline failure alert."""
        alert = Alert(
            title=f"Pipeline Failed: {pipeline_name}",
            message=f"Pipeline '{pipeline_name}' failed at stage '{stage_name}'.\n\n"
                    f"Error: {error_message}",
            severity=AlertSeverity.CRITICAL,
            source="ETL Pipeline",
            timestamp=datetime.now(),
            metadata={
                "pipeline_name": pipeline_name,
                "failed_stage": stage_name,
                "error": error_message
            }
        )
        
        return self.send_alert(alert)
    
    def send_data_quality_alert(
        self,
        table_name: str,
        quality_score: float,
        failed_checks: List[str]
    ) -> bool:
        """Send data quality alert."""
        severity = AlertSeverity.WARNING if quality_score >= 0.9 else AlertSeverity.ERROR
        
        alert = Alert(
            title=f"Data Quality Issue: {table_name}",
            message=f"Data quality check failed for table '{table_name}'.\n\n"
                    f"Quality Score: {quality_score:.2%}\n"
                    f"Failed Checks:\n" + "\n".join(f"  - {c}" for c in failed_checks),
            severity=severity,
            source="Data Quality",
            timestamp=datetime.now(),
            metadata={
                "table_name": table_name,
                "quality_score": quality_score,
                "failed_checks": failed_checks
            }
        )
        
        return self.send_alert(alert)
    
    def send_pipeline_success_alert(
        self,
        pipeline_name: str,
        duration_seconds: float,
        rows_processed: int
    ) -> bool:
        """Send pipeline success notification."""
        alert = Alert(
            title=f"Pipeline Completed: {pipeline_name}",
            message=f"Pipeline '{pipeline_name}' completed successfully.\n\n"
                    f"Duration: {duration_seconds:.2f} seconds\n"
                    f"Rows Processed: {rows_processed:,}",
            severity=AlertSeverity.INFO,
            source="ETL Pipeline",
            timestamp=datetime.now(),
            metadata={
                "pipeline_name": pipeline_name,
                "duration_seconds": duration_seconds,
                "rows_processed": rows_processed
            }
        )
        
        return self.send_alert(alert)


class BaseAlerter:
    """Base class for alerting channels."""
    
    def send(self, alert: Alert) -> bool:
        """Send an alert. Override in subclasses."""
        raise NotImplementedError


class SNSAlerter(BaseAlerter):
    """
    AWS SNS alerter.
    Publishes alerts to SNS topics for email/SMS notifications.
    """
    
    def __init__(
        self,
        topic_name: str = None,
        topic_arn: str = None,
        region: str = "us-east-1"
    ):
        self.topic_name = topic_name
        self.topic_arn = topic_arn
        self.region = region
        self._client = None
    
    @property
    def client(self):
        """Lazy-load SNS client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('sns', region_name=self.region)
            except ImportError:
                logger.warning("boto3 not available. SNS alerting disabled.")
        return self._client
    
    def _get_topic_arn(self) -> Optional[str]:
        """Get or create SNS topic ARN."""
        if self.topic_arn:
            return self.topic_arn
        
        if self.client is None:
            return None
        
        try:
            # List topics to find existing one
            response = self.client.list_topics()
            
            for topic in response.get('Topics', []):
                if self.topic_name in topic['TopicArn']:
                    self.topic_arn = topic['TopicArn']
                    return self.topic_arn
            
            # Create topic if not found
            response = self.client.create_topic(Name=self.topic_name)
            self.topic_arn = response['TopicArn']
            logger.info(f"Created SNS topic: {self.topic_arn}")
            
            return self.topic_arn
            
        except Exception as e:
            logger.error(f"Failed to get/create SNS topic: {e}")
            return None
    
    def send(self, alert: Alert) -> bool:
        """Publish alert to SNS topic."""
        topic_arn = self._get_topic_arn()
        
        if topic_arn is None:
            logger.warning("No SNS topic available. Alert not sent.")
            return False
        
        try:
            response = self.client.publish(
                TopicArn=topic_arn,
                Subject=f"[{alert.severity.value.upper()}] {alert.title}",
                Message=alert.to_sns_message()
            )
            
            logger.info(f"Published alert to SNS: {response['MessageId']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish SNS alert: {e}")
            return False
    
    def subscribe_email(self, email: str) -> bool:
        """Subscribe email to alert topic."""
        topic_arn = self._get_topic_arn()
        
        if topic_arn is None:
            return False
        
        try:
            self.client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            
            logger.info(f"Subscribed {email} to alerts. Check email for confirmation.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe email: {e}")
            return False


class FileAlerter(BaseAlerter):
    """
    File-based alerter.
    Writes alerts to local JSON files for logging and audit.
    """
    
    def __init__(self, alerts_dir: str = "logs/alerts"):
        self.alerts_dir = alerts_dir
        os.makedirs(alerts_dir, exist_ok=True)
    
    def send(self, alert: Alert) -> bool:
        """Write alert to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alert_{alert.severity.value}_{timestamp}.json"
            filepath = os.path.join(self.alerts_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
            
            logger.debug(f"Alert saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write alert file: {e}")
            return False


class SlackAlerter(BaseAlerter):
    """
    Slack alerter for team notifications.
    Uses Slack webhooks for message delivery.
    """
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            import requests
            
            # Format message for Slack
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.ERROR: "#f44336",
                AlertSeverity.CRITICAL: "#9c27b0"
            }.get(alert.severity, "#000000")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "Claims Pipeline Alerting"
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
