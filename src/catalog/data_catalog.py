"""
Data Catalog Module
Manages data asset metadata, documentation, and discovery.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AssetType(Enum):
    """Types of data assets."""
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    STREAM = "stream"
    API = "api"


class DataLayer(Enum):
    """Data layers in medallion architecture."""
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class DataAsset:
    """Data asset metadata."""
    id: str
    name: str
    asset_type: AssetType
    layer: DataLayer
    description: str
    
    # Location
    location: str
    format: str = "parquet"
    
    # Schema
    schema: Dict[str, Any] = field(default_factory=dict)
    
    # Quality
    quality_score: Optional[float] = None
    last_quality_check: Optional[datetime] = None
    
    # Ownership
    owner: str = "data-engineering"
    steward: str = ""
    
    # Classification
    tags: List[str] = field(default_factory=list)
    domain: str = "healthcare"
    pii_columns: List[str] = field(default_factory=list)
    
    # Lineage
    upstream_assets: List[str] = field(default_factory=list)
    downstream_assets: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    # Statistics
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["asset_type"] = self.asset_type.value
        result["layer"] = self.layer.value
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        if self.last_quality_check:
            result["last_quality_check"] = self.last_quality_check.isoformat()
        if self.last_updated:
            result["last_updated"] = self.last_updated.isoformat()
        return result


class DataCatalog:
    """
    Central data catalog for asset discovery and governance.
    """
    
    def __init__(
        self,
        catalog_path: str = "data/catalog",
        use_glue: bool = False,
        aws_config: Optional[Dict[str, Any]] = None
    ):
        self.catalog_path = catalog_path
        self.use_glue = use_glue
        self.aws_config = aws_config or {}
        self.assets: Dict[str, DataAsset] = {}
        
        os.makedirs(catalog_path, exist_ok=True)
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load catalog from storage."""
        catalog_file = os.path.join(self.catalog_path, "catalog.json")
        
        if os.path.exists(catalog_file):
            with open(catalog_file, 'r') as f:
                data = json.load(f)
                for asset_id, asset_data in data.get("assets", {}).items():
                    # Convert enums back
                    asset_data["asset_type"] = AssetType(asset_data["asset_type"])
                    asset_data["layer"] = DataLayer(asset_data["layer"])
                    asset_data["created_at"] = datetime.fromisoformat(asset_data["created_at"])
                    asset_data["updated_at"] = datetime.fromisoformat(asset_data["updated_at"])
                    
                    self.assets[asset_id] = DataAsset(**asset_data)
            
            logger.info(f"Loaded {len(self.assets)} assets from catalog")
    
    def _save_catalog(self) -> None:
        """Save catalog to storage."""
        catalog_file = os.path.join(self.catalog_path, "catalog.json")
        
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "assets": {
                asset_id: asset.to_dict()
                for asset_id, asset in self.assets.items()
            }
        }
        
        with open(catalog_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.assets)} assets to catalog")
    
    def register_asset(self, asset: DataAsset) -> str:
        """
        Register a data asset in the catalog.
        
        Args:
            asset: DataAsset to register
            
        Returns:
            Asset ID
        """
        asset.updated_at = datetime.now()
        self.assets[asset.id] = asset
        self._save_catalog()
        
        logger.info(f"Registered asset: {asset.name} ({asset.id})")
        
        # Sync to Glue if enabled
        if self.use_glue:
            self._sync_to_glue(asset)
        
        return asset.id
    
    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """Get asset by ID."""
        return self.assets.get(asset_id)
    
    def search_assets(
        self,
        name: Optional[str] = None,
        layer: Optional[DataLayer] = None,
        tags: Optional[List[str]] = None,
        domain: Optional[str] = None
    ) -> List[DataAsset]:
        """
        Search for assets matching criteria.
        
        Args:
            name: Name pattern to match
            layer: Data layer filter
            tags: Tags to filter by
            domain: Domain filter
            
        Returns:
            List of matching assets
        """
        results = []
        
        for asset in self.assets.values():
            # Filter by name
            if name and name.lower() not in asset.name.lower():
                continue
            
            # Filter by layer
            if layer and asset.layer != layer:
                continue
            
            # Filter by tags
            if tags:
                if not any(tag in asset.tags for tag in tags):
                    continue
            
            # Filter by domain
            if domain and asset.domain != domain:
                continue
            
            results.append(asset)
        
        return results
    
    def get_lineage(self, asset_id: str) -> Dict[str, Any]:
        """
        Get lineage information for an asset.
        
        Args:
            asset_id: Asset ID
            
        Returns:
            Lineage graph
        """
        asset = self.get_asset(asset_id)
        if not asset:
            return {}
        
        return {
            "asset_id": asset_id,
            "name": asset.name,
            "upstream": [
                self.get_asset(uid).to_dict() if self.get_asset(uid) else {"id": uid}
                for uid in asset.upstream_assets
            ],
            "downstream": [
                self.get_asset(did).to_dict() if self.get_asset(did) else {"id": did}
                for did in asset.downstream_assets
            ]
        }
    
    def update_statistics(
        self,
        asset_id: str,
        row_count: int,
        size_bytes: int
    ) -> None:
        """Update asset statistics."""
        asset = self.get_asset(asset_id)
        if asset:
            asset.row_count = row_count
            asset.size_bytes = size_bytes
            asset.last_updated = datetime.now()
            asset.updated_at = datetime.now()
            self._save_catalog()
    
    def update_quality_score(
        self,
        asset_id: str,
        score: float
    ) -> None:
        """Update asset quality score."""
        asset = self.get_asset(asset_id)
        if asset:
            asset.quality_score = score
            asset.last_quality_check = datetime.now()
            asset.updated_at = datetime.now()
            self._save_catalog()
    
    def _sync_to_glue(self, asset: DataAsset) -> None:
        """Sync asset to AWS Glue Data Catalog."""
        try:
            import boto3
            
            glue = boto3.client(
                'glue',
                region_name=self.aws_config.get('region', 'us-east-1')
            )
            
            database = self.aws_config.get('glue', {}).get('database', 'claims_catalog')
            
            # Create or update table
            table_input = {
                'Name': asset.name,
                'Description': asset.description,
                'StorageDescriptor': {
                    'Location': asset.location,
                    'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    },
                    'Columns': [
                        {'Name': col_name, 'Type': col_type}
                        for col_name, col_type in asset.schema.items()
                    ]
                },
                'Parameters': {
                    'layer': asset.layer.value,
                    'owner': asset.owner,
                    'domain': asset.domain
                }
            }
            
            try:
                glue.create_table(
                    DatabaseName=database,
                    TableInput=table_input
                )
                logger.info(f"Created Glue table: {asset.name}")
            except glue.exceptions.AlreadyExistsException:
                glue.update_table(
                    DatabaseName=database,
                    TableInput=table_input
                )
                logger.info(f"Updated Glue table: {asset.name}")
                
        except ImportError:
            logger.warning("boto3 not available. Skipping Glue sync.")
        except Exception as e:
            logger.error(f"Failed to sync to Glue: {e}")
    
    def generate_documentation(self, output_path: str = "docs/data_catalog") -> str:
        """
        Generate documentation for all assets.
        
        Args:
            output_path: Output directory
            
        Returns:
            Path to generated documentation
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Generate index
        index_content = """# Data Catalog

## Overview

This catalog contains metadata for all data assets in the claims pipeline.

## Assets by Layer

"""
        
        # Group by layer
        by_layer = {}
        for asset in self.assets.values():
            layer = asset.layer.value
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(asset)
        
        for layer, assets in sorted(by_layer.items()):
            index_content += f"\n### {layer.title()} Layer\n\n"
            for asset in assets:
                index_content += f"- [{asset.name}](./{asset.id}.md) - {asset.description[:50]}...\n"
        
        # Write index
        index_path = os.path.join(output_path, "index.md")
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        # Generate individual asset docs
        for asset in self.assets.values():
            asset_doc = f"""# {asset.name}

## Overview

- **ID**: {asset.id}
- **Type**: {asset.asset_type.value}
- **Layer**: {asset.layer.value}
- **Location**: {asset.location}
- **Format**: {asset.format}

## Description

{asset.description}

## Ownership

- **Owner**: {asset.owner}
- **Steward**: {asset.steward or 'N/A'}
- **Domain**: {asset.domain}

## Statistics

- **Row Count**: {asset.row_count:,} if asset.row_count else 'N/A'
- **Size**: {asset.size_bytes / (1024*1024):.2f} MB if asset.size_bytes else 'N/A'
- **Last Updated**: {asset.last_updated or 'N/A'}

## Quality

- **Quality Score**: {asset.quality_score:.2%} if asset.quality_score else 'N/A'
- **Last Check**: {asset.last_quality_check or 'N/A'}

## Schema

| Column | Type |
|--------|------|
"""
            for col_name, col_type in asset.schema.items():
                asset_doc += f"| {col_name} | {col_type} |\n"
            
            asset_doc += f"""
## Lineage

### Upstream
{chr(10).join(f'- {uid}' for uid in asset.upstream_assets) or 'None'}

### Downstream
{chr(10).join(f'- {did}' for did in asset.downstream_assets) or 'None'}

## Tags

{', '.join(asset.tags) or 'None'}

## PII Columns

{', '.join(asset.pii_columns) or 'None'}

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            asset_path = os.path.join(output_path, f"{asset.id}.md")
            with open(asset_path, 'w') as f:
                f.write(asset_doc)
        
        logger.info(f"Generated documentation for {len(self.assets)} assets")
        return output_path
