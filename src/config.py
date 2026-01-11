"""
Configuration Management Module
Handles loading and validation of configuration settings.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data-related configuration settings."""
    raw_data_file: str
    delimiter: str
    total_rows: int
    chunk_size: int
    sample_size: int
    random_seed: int
    target_column: str
    id_column: str
    high_missing_threshold: int
    columns_to_remove: List[str]
    final_columns_to_drop: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]


@dataclass
class S3Config:
    """AWS S3 configuration settings."""
    bucket_name: str
    raw_data_prefix: str
    processed_data_prefix: str
    models_prefix: str
    artifacts_prefix: str


@dataclass
class RedshiftConfig:
    """AWS Redshift configuration settings."""
    cluster_identifier: str
    database: str
    schema: str
    port: int


@dataclass
class GlueConfig:
    """AWS Glue configuration settings."""
    database: str
    crawler_name: str
    job_name: str


@dataclass
class LambdaConfig:
    """AWS Lambda configuration settings."""
    function_name: str
    timeout: int
    memory_size: int


@dataclass
class AWSConfig:
    """AWS configuration settings."""
    region: str
    s3: S3Config
    redshift: RedshiftConfig
    glue: GlueConfig
    lambda_config: LambdaConfig


@dataclass
class ModelHyperparameters:
    """Model hyperparameters configuration."""
    lasso: Dict[str, Any]
    ridge: Dict[str, Any]
    random_forest: Dict[str, Any]
    mars: Dict[str, Any]
    adaboost: Dict[str, Any]
    xgboost: Dict[str, Any]
    lightgbm: Dict[str, Any]


@dataclass
class ModelConfig:
    """Model training configuration settings."""
    test_size: float
    validation_size: float
    random_state: int
    hyperparameters: ModelHyperparameters


@dataclass
class PathsConfig:
    """Project paths configuration."""
    data_dir: str
    raw_dir: str
    processed_dir: str
    interim_dir: str
    external_dir: str
    models_dir: str
    logs_dir: str
    reports_dir: str
    figures_dir: str


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    file: str


@dataclass
class ProjectConfig:
    """Complete project configuration."""
    name: str
    version: str
    description: str
    data: DataConfig
    aws: AWSConfig
    model: ModelConfig
    paths: PathsConfig
    logging: LoggingConfig


class ConfigurationManager:
    """
    Manages configuration loading, validation, and access.
    Implements singleton pattern for global config access.
    """
    
    _instance = None
    _config: Optional[ProjectConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._config is None:
            if config_path is None:
                # Default config path
                config_path = self._find_config_path()
            self.load_config(config_path)
    
    def _find_config_path(self) -> str:
        """Find the configuration file path."""
        # Try multiple locations
        possible_paths = [
            Path("config/settings.yaml"),
            Path("../config/settings.yaml"),
            Path(__file__).parent.parent / "config" / "settings.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "Configuration file not found. Please ensure 'config/settings.yaml' exists."
        )
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        self._config = self._parse_config(raw_config)
        self._create_directories()
    
    def _parse_config(self, raw: Dict[str, Any]) -> ProjectConfig:
        """Parse raw configuration dictionary into typed dataclasses."""
        
        # Parse data config
        data_config = DataConfig(
            raw_data_file=raw['data']['raw_data_file'],
            delimiter=raw['data']['delimiter'],
            total_rows=raw['data']['total_rows'],
            chunk_size=raw['data']['chunk_size'],
            sample_size=raw['data']['sample_size'],
            random_seed=raw['data']['random_seed'],
            target_column=raw['data']['target_column'],
            id_column=raw['data']['id_column'],
            high_missing_threshold=raw['data']['high_missing_threshold'],
            columns_to_remove=raw['data']['columns_to_remove'],
            final_columns_to_drop=raw['data']['final_columns_to_drop'],
            categorical_columns=raw['data']['categorical_columns'],
            numerical_columns=raw['data']['numerical_columns'],
        )
        
        # Parse AWS config
        s3_config = S3Config(**raw['aws']['s3'])
        redshift_config = RedshiftConfig(**raw['aws']['redshift'])
        glue_config = GlueConfig(**raw['aws']['glue'])
        lambda_config = LambdaConfig(**raw['aws']['lambda'])
        
        aws_config = AWSConfig(
            region=raw['aws']['region'],
            s3=s3_config,
            redshift=redshift_config,
            glue=glue_config,
            lambda_config=lambda_config,
        )
        
        # Parse model config
        hyperparams = ModelHyperparameters(
            lasso=raw['model']['lasso'],
            ridge=raw['model']['ridge'],
            random_forest=raw['model']['random_forest'],
            mars=raw['model']['mars'],
            adaboost=raw['model']['adaboost'],
            xgboost=raw['model']['xgboost'],
            lightgbm=raw['model']['lightgbm'],
        )
        
        model_config = ModelConfig(
            test_size=raw['model']['test_size'],
            validation_size=raw['model']['validation_size'],
            random_state=raw['model']['random_state'],
            hyperparameters=hyperparams,
        )
        
        # Parse paths config
        paths_config = PathsConfig(**raw['paths'])
        
        # Parse logging config
        logging_config = LoggingConfig(**raw['logging'])
        
        return ProjectConfig(
            name=raw['project']['name'],
            version=raw['project']['version'],
            description=raw['project']['description'],
            data=data_config,
            aws=aws_config,
            model=model_config,
            paths=paths_config,
            logging=logging_config,
        )
    
    def _create_directories(self) -> None:
        """Create necessary project directories."""
        if self._config is None:
            return
        
        dirs = [
            self._config.paths.data_dir,
            self._config.paths.raw_dir,
            self._config.paths.processed_dir,
            self._config.paths.interim_dir,
            self._config.paths.external_dir,
            self._config.paths.models_dir,
            self._config.paths.logs_dir,
            self._config.paths.reports_dir,
            self._config.paths.figures_dir,
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> ProjectConfig:
        """Get the loaded configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.config.data
    
    def get_aws_config(self) -> AWSConfig:
        """Get AWS configuration."""
        return self.config.aws
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config.model
    
    def get_paths_config(self) -> PathsConfig:
        """Get paths configuration."""
        return self.config.paths


def get_config() -> ProjectConfig:
    """Convenience function to get configuration."""
    return ConfigurationManager().config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
