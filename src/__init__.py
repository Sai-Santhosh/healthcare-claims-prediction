# Medical Claims Data Engineering Platform
# =============================================================================
# Production-grade ETL/ELT pipeline for healthcare claims data processing
# =============================================================================

__version__ = "2.0.0"
__author__ = "Data Engineering Team"

from . import etl
from . import data_quality
from . import monitoring
from . import catalog
from . import aws
from . import utils

__all__ = [
    'etl',
    'data_quality',
    'monitoring',
    'catalog',
    'aws',
    'utils'
]
