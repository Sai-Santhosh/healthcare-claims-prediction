"""
Dashboard Generation Module
Creates CloudWatch dashboards and generates HTML reports.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DashboardGenerator:
    """
    Generates monitoring dashboards for pipeline visibility.
    Supports CloudWatch dashboards and HTML reports.
    """
    
    def __init__(
        self,
        namespace: str = "ClaimsPipeline",
        region: str = "us-east-1",
        reports_dir: str = "reports/dashboards"
    ):
        self.namespace = namespace
        self.region = region
        self.reports_dir = reports_dir
        self._client = None
        
        os.makedirs(reports_dir, exist_ok=True)
    
    @property
    def client(self):
        """Lazy-load CloudWatch client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('cloudwatch', region_name=self.region)
            except ImportError:
                logger.warning("boto3 not available. Dashboard features limited.")
        return self._client
    
    def create_cloudwatch_dashboard(
        self,
        dashboard_name: str = "ClaimsPipelineDashboard"
    ) -> bool:
        """
        Create CloudWatch dashboard for pipeline monitoring.
        
        Args:
            dashboard_name: Name for the dashboard
            
        Returns:
            True if successful
        """
        if self.client is None:
            logger.warning("Cannot create CloudWatch dashboard without boto3")
            return False
        
        dashboard_body = {
            "widgets": [
                # Pipeline Success/Failure
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Pipeline Executions",
                        "region": self.region,
                        "metrics": [
                            [self.namespace, "PipelineSuccess", {"color": "#2ca02c"}],
                            [self.namespace, "PipelineFailure", {"color": "#d62728"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "period": 300,
                        "stat": "Sum"
                    }
                },
                # Rows Processed
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Rows Processed",
                        "region": self.region,
                        "metrics": [
                            [self.namespace, "RowsProcessed"]
                        ],
                        "view": "timeSeries",
                        "period": 300,
                        "stat": "Sum"
                    }
                },
                # Pipeline Duration
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Pipeline Duration (seconds)",
                        "region": self.region,
                        "metrics": [
                            [self.namespace, "PipelineDuration"]
                        ],
                        "view": "timeSeries",
                        "period": 300,
                        "stat": "Average"
                    }
                },
                # Data Quality Score
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Data Quality Score",
                        "region": self.region,
                        "metrics": [
                            [self.namespace, "DataQualityScore"]
                        ],
                        "view": "timeSeries",
                        "period": 300,
                        "stat": "Average"
                    }
                },
                # Stage Performance
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "title": "Stage Duration by Stage",
                        "region": self.region,
                        "metrics": [
                            [self.namespace, "StageDuration", "StageName", "extract"],
                            [self.namespace, "StageDuration", "StageName", "transform"],
                            [self.namespace, "StageDuration", "StageName", "load"],
                            [self.namespace, "StageDuration", "StageName", "quality_check"]
                        ],
                        "view": "timeSeries",
                        "period": 300,
                        "stat": "Average"
                    }
                }
            ]
        }
        
        try:
            self.client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info(f"Created CloudWatch dashboard: {dashboard_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def generate_html_report(
        self,
        pipeline_results: List[Dict[str, Any]],
        report_name: str = "pipeline_report"
    ) -> str:
        """
        Generate HTML dashboard report.
        
        Args:
            pipeline_results: List of pipeline execution results
            report_name: Name for the report file
            
        Returns:
            Path to generated HTML file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.reports_dir, f"{report_name}_{timestamp}.html")
        
        # Calculate summary statistics
        total_runs = len(pipeline_results)
        successful_runs = sum(1 for r in pipeline_results if r.get('status') == 'success')
        failed_runs = total_runs - successful_runs
        total_rows = sum(r.get('total_rows_processed', 0) for r in pipeline_results)
        avg_duration = sum(r.get('total_duration_seconds', 0) for r in pipeline_results) / max(total_runs, 1)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Monitoring Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        .timestamp {{
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.success {{
            color: #27ae60;
        }}
        .metric-value.error {{
            color: #e74c3c;
        }}
        .metric-trend {{
            font-size: 0.8rem;
            color: #95a5a6;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f5f6fa;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            color: #7f8c8d;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .status-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-badge.failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .progress-bar {{
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Claims Pipeline Monitoring Dashboard</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Runs</h3>
                <div class="metric-value">{total_runs}</div>
                <div class="metric-trend">Last 24 hours</div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value success">{successful_runs / max(total_runs, 1) * 100:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {successful_runs / max(total_runs, 1) * 100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <h3>Rows Processed</h3>
                <div class="metric-value">{total_rows:,}</div>
                <div class="metric-trend">Cumulative</div>
            </div>
            <div class="metric-card">
                <h3>Avg Duration</h3>
                <div class="metric-value">{avg_duration:.1f}s</div>
                <div class="metric-trend">Per execution</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Recent Pipeline Executions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pipeline ID</th>
                        <th>Status</th>
                        <th>Start Time</th>
                        <th>Duration</th>
                        <th>Rows</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._generate_table_rows(pipeline_results))}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {filepath}")
        return filepath
    
    def _generate_table_rows(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate HTML table rows for pipeline results."""
        rows = []
        
        for result in results[-20:]:  # Last 20 executions
            status_class = "success" if result.get('status') == 'success' else "failed"
            
            row = f"""
                <tr>
                    <td>{result.get('pipeline_id', 'N/A')[:8]}...</td>
                    <td><span class="status-badge {status_class}">{result.get('status', 'unknown')}</span></td>
                    <td>{result.get('start_time', 'N/A')}</td>
                    <td>{result.get('total_duration_seconds', 0):.2f}s</td>
                    <td>{result.get('total_rows_processed', 0):,}</td>
                </tr>
            """
            rows.append(row)
        
        return rows
