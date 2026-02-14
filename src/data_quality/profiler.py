"""
Data Profiling Module
Generates comprehensive data profiles for analysis and documentation.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric stats (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical stats (if applicable)
    top_values: Optional[List[Dict[str, Any]]] = None
    
    # Additional info
    memory_bytes: int = 0
    sample_values: Optional[List[Any]] = None


@dataclass
class DataProfile:
    """Complete profile for a dataset."""
    name: str
    profile_time: datetime
    row_count: int
    column_count: int
    memory_mb: float
    duplicate_rows: int
    duplicate_percentage: float
    columns: List[ColumnProfile] = field(default_factory=list)
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "profile_time": self.profile_time.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_mb": self.memory_mb,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_percentage": self.duplicate_percentage,
            "columns": [asdict(col) for col in self.columns],
            "correlations": self.correlations
        }


class DataProfiler:
    """
    Generates comprehensive data profiles.
    """
    
    def __init__(
        self,
        sample_size: int = 100000,
        top_n_values: int = 10,
        reports_dir: str = "reports/profiles"
    ):
        self.sample_size = sample_size
        self.top_n_values = top_n_values
        self.reports_dir = reports_dir
        
        os.makedirs(reports_dir, exist_ok=True)
    
    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataset"
    ) -> DataProfile:
        """
        Generate comprehensive profile for a DataFrame.
        
        Args:
            df: DataFrame to profile
            name: Name for the dataset
            
        Returns:
            DataProfile with all statistics
        """
        logger.info(f"Profiling dataset: {name} ({len(df):,} rows, {len(df.columns)} columns)")
        
        # Sample for large datasets
        if len(df) > self.sample_size:
            logger.info(f"Sampling {self.sample_size:,} rows for profiling")
            df_sample = df.sample(n=self.sample_size, random_state=42)
        else:
            df_sample = df
        
        # Basic stats
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        duplicate_rows = df.duplicated().sum()
        
        # Profile columns
        columns = []
        for col in df.columns:
            col_profile = self._profile_column(df[col], df_sample[col], col)
            columns.append(col_profile)
        
        # Compute correlations for numeric columns
        correlations = self._compute_correlations(df_sample)
        
        profile = DataProfile(
            name=name,
            profile_time=datetime.now(),
            row_count=len(df),
            column_count=len(df.columns),
            memory_mb=round(memory_mb, 2),
            duplicate_rows=duplicate_rows,
            duplicate_percentage=round(duplicate_rows / max(len(df), 1) * 100, 2),
            columns=columns,
            correlations=correlations
        )
        
        return profile
    
    def _profile_column(
        self,
        series: pd.Series,
        sample_series: pd.Series,
        name: str
    ) -> ColumnProfile:
        """Profile a single column."""
        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            count=len(series),
            missing_count=int(series.isnull().sum()),
            missing_percentage=round(series.isnull().mean() * 100, 2),
            unique_count=int(series.nunique()),
            unique_percentage=round(series.nunique() / max(len(series), 1) * 100, 2),
            memory_bytes=int(series.memory_usage(deep=True))
        )
        
        # Sample values
        non_null = series.dropna()
        if len(non_null) > 0:
            profile.sample_values = non_null.head(5).tolist()
        
        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            profile.mean = round(float(series.mean()), 4) if not series.isnull().all() else None
            profile.std = round(float(series.std()), 4) if not series.isnull().all() else None
            profile.min = float(series.min()) if not series.isnull().all() else None
            profile.max = float(series.max()) if not series.isnull().all() else None
            profile.median = round(float(series.median()), 4) if not series.isnull().all() else None
            profile.q1 = round(float(series.quantile(0.25)), 4) if not series.isnull().all() else None
            profile.q3 = round(float(series.quantile(0.75)), 4) if not series.isnull().all() else None
            
            try:
                profile.skewness = round(float(series.skew()), 4)
                profile.kurtosis = round(float(series.kurtosis()), 4)
            except:
                pass
        
        # Top values (for all columns)
        value_counts = sample_series.value_counts().head(self.top_n_values)
        profile.top_values = [
            {"value": str(val), "count": int(count), "percentage": round(count / len(sample_series) * 100, 2)}
            for val, count in value_counts.items()
        ]
        
        return profile
    
    def _compute_correlations(
        self,
        df: pd.DataFrame
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Compute correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Convert to nested dict
            correlations = {}
            for col1 in corr_matrix.columns:
                correlations[col1] = {}
                for col2 in corr_matrix.columns:
                    correlations[col1][col2] = round(corr_matrix.loc[col1, col2], 4)
            
            return correlations
            
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")
            return None
    
    def save_profile(
        self,
        profile: DataProfile,
        format: str = "json"
    ) -> str:
        """
        Save profile to file.
        
        Args:
            profile: DataProfile to save
            format: Output format ('json' or 'html')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "json":
            filepath = os.path.join(self.reports_dir, f"profile_{profile.name}_{timestamp}.json")
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2, default=str)
        
        elif format == "html":
            filepath = os.path.join(self.reports_dir, f"profile_{profile.name}_{timestamp}.html")
            html_content = self._generate_html_report(profile)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved profile to {filepath}")
        return filepath
    
    def _generate_html_report(self, profile: DataProfile) -> str:
        """Generate HTML profile report."""
        # Generate column rows
        column_rows = ""
        for col in profile.columns:
            column_rows += f"""
                <tr>
                    <td><strong>{col.name}</strong></td>
                    <td><code>{col.dtype}</code></td>
                    <td>{col.count:,}</td>
                    <td class="{'warning' if col.missing_percentage > 10 else ''}">{col.missing_count:,} ({col.missing_percentage}%)</td>
                    <td>{col.unique_count:,}</td>
                    <td>{col.mean if col.mean else '-'}</td>
                    <td>{col.std if col.std else '-'}</td>
                    <td>{col.min if col.min is not None else '-'}</td>
                    <td>{col.max if col.max is not None else '-'}</td>
                </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Profile: {profile.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            font-size: 0.8rem;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .stat-card .value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h2 {{
            font-size: 1.3rem;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            color: #7f8c8d;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        code {{
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85rem;
        }}
        .warning {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Data Profile: {profile.name}</h1>
            <p>Generated: {profile.profile_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Rows</h3>
                <div class="value">{profile.row_count:,}</div>
            </div>
            <div class="stat-card">
                <h3>Columns</h3>
                <div class="value">{profile.column_count}</div>
            </div>
            <div class="stat-card">
                <h3>Memory</h3>
                <div class="value">{profile.memory_mb:.1f} MB</div>
            </div>
            <div class="stat-card">
                <h3>Duplicates</h3>
                <div class="value">{profile.duplicate_percentage}%</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Column Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Missing</th>
                        <th>Unique</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    {column_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def compare_profiles(
        self,
        profile1: DataProfile,
        profile2: DataProfile
    ) -> Dict[str, Any]:
        """
        Compare two data profiles.
        Useful for detecting data drift.
        
        Args:
            profile1: First profile (baseline)
            profile2: Second profile (current)
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "row_count_change": profile2.row_count - profile1.row_count,
            "row_count_pct_change": (profile2.row_count - profile1.row_count) / max(profile1.row_count, 1) * 100,
            "column_changes": {
                "added": [],
                "removed": [],
                "modified": []
            }
        }
        
        cols1 = {c.name: c for c in profile1.columns}
        cols2 = {c.name: c for c in profile2.columns}
        
        # Find added columns
        comparison["column_changes"]["added"] = list(set(cols2.keys()) - set(cols1.keys()))
        
        # Find removed columns
        comparison["column_changes"]["removed"] = list(set(cols1.keys()) - set(cols2.keys()))
        
        # Find modified columns
        for name in set(cols1.keys()) & set(cols2.keys()):
            c1, c2 = cols1[name], cols2[name]
            
            changes = {}
            
            # Check dtype change
            if c1.dtype != c2.dtype:
                changes["dtype"] = {"old": c1.dtype, "new": c2.dtype}
            
            # Check significant mean change for numeric columns
            if c1.mean is not None and c2.mean is not None:
                mean_change = abs(c2.mean - c1.mean) / max(abs(c1.mean), 1)
                if mean_change > 0.1:  # >10% change
                    changes["mean"] = {"old": c1.mean, "new": c2.mean, "pct_change": mean_change * 100}
            
            # Check missing percentage change
            if abs(c2.missing_percentage - c1.missing_percentage) > 5:  # >5% change
                changes["missing_percentage"] = {
                    "old": c1.missing_percentage,
                    "new": c2.missing_percentage
                }
            
            if changes:
                comparison["column_changes"]["modified"].append({
                    "column": name,
                    "changes": changes
                })
        
        return comparison
