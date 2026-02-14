"""
Data Lineage Tracking Module
Tracks data transformations and dependencies.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class OperationType(Enum):
    """Types of data operations."""
    READ = "read"
    WRITE = "write"
    TRANSFORM = "transform"
    JOIN = "join"
    AGGREGATE = "aggregate"
    FILTER = "filter"


@dataclass
class LineageNode:
    """Node in the lineage graph."""
    id: str
    name: str
    node_type: str  # source, transformation, target
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """Edge in the lineage graph representing data flow."""
    source_id: str
    target_id: str
    operation: OperationType
    columns_affected: List[str] = field(default_factory=list)
    transformation_logic: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LineageRecord:
    """Record of a lineage event."""
    job_id: str
    job_name: str
    start_time: datetime
    end_time: Optional[datetime]
    inputs: List[str]
    outputs: List[str]
    operations: List[Dict[str, Any]]
    status: str = "running"
    error: Optional[str] = None


class LineageTracker:
    """
    Tracks data lineage across pipeline stages.
    """
    
    def __init__(
        self,
        lineage_path: str = "data/lineage",
        enable_column_lineage: bool = True
    ):
        self.lineage_path = lineage_path
        self.enable_column_lineage = enable_column_lineage
        
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self.records: List[LineageRecord] = []
        self._current_record: Optional[LineageRecord] = None
        
        os.makedirs(lineage_path, exist_ok=True)
        self._load_lineage()
    
    def _load_lineage(self) -> None:
        """Load lineage from storage."""
        lineage_file = os.path.join(self.lineage_path, "lineage.json")
        
        if os.path.exists(lineage_file):
            with open(lineage_file, 'r') as f:
                data = json.load(f)
                
                # Load nodes
                for node_data in data.get("nodes", []):
                    node = LineageNode(**node_data)
                    self.nodes[node.id] = node
                
                # Load edges
                for edge_data in data.get("edges", []):
                    edge_data["operation"] = OperationType(edge_data["operation"])
                    edge_data["timestamp"] = datetime.fromisoformat(edge_data["timestamp"])
                    self.edges.append(LineageEdge(**edge_data))
            
            logger.info(f"Loaded lineage: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _save_lineage(self) -> None:
        """Save lineage to storage."""
        lineage_file = os.path.join(self.lineage_path, "lineage.json")
        
        data = {
            "updated_at": datetime.now().isoformat(),
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "operation": edge.operation.value,
                    "columns_affected": edge.columns_affected,
                    "transformation_logic": edge.transformation_logic,
                    "timestamp": edge.timestamp.isoformat()
                }
                for edge in self.edges
            ]
        }
        
        with open(lineage_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def start_job(self, job_id: str, job_name: str) -> None:
        """Start tracking a new job."""
        self._current_record = LineageRecord(
            job_id=job_id,
            job_name=job_name,
            start_time=datetime.now(),
            end_time=None,
            inputs=[],
            outputs=[],
            operations=[]
        )
        logger.info(f"Started lineage tracking for job: {job_name}")
    
    def end_job(self, status: str = "success", error: Optional[str] = None) -> None:
        """End tracking for current job."""
        if self._current_record:
            self._current_record.end_time = datetime.now()
            self._current_record.status = status
            self._current_record.error = error
            
            self.records.append(self._current_record)
            self._save_job_record(self._current_record)
            
            logger.info(f"Ended lineage tracking: {status}")
            self._current_record = None
    
    def _save_job_record(self, record: LineageRecord) -> None:
        """Save job lineage record."""
        record_file = os.path.join(
            self.lineage_path,
            "jobs",
            f"{record.job_id}.json"
        )
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        
        with open(record_file, 'w') as f:
            json.dump({
                "job_id": record.job_id,
                "job_name": record.job_name,
                "start_time": record.start_time.isoformat(),
                "end_time": record.end_time.isoformat() if record.end_time else None,
                "inputs": record.inputs,
                "outputs": record.outputs,
                "operations": record.operations,
                "status": record.status,
                "error": record.error
            }, f, indent=2)
    
    def register_source(
        self,
        source_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a data source."""
        node = LineageNode(
            id=source_id,
            name=name,
            node_type="source",
            metadata=metadata or {}
        )
        self.nodes[source_id] = node
        
        if self._current_record:
            self._current_record.inputs.append(source_id)
        
        self._save_lineage()
        logger.debug(f"Registered source: {name}")
    
    def register_target(
        self,
        target_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a data target."""
        node = LineageNode(
            id=target_id,
            name=name,
            node_type="target",
            metadata=metadata or {}
        )
        self.nodes[target_id] = node
        
        if self._current_record:
            self._current_record.outputs.append(target_id)
        
        self._save_lineage()
        logger.debug(f"Registered target: {name}")
    
    def track_transformation(
        self,
        source_ids: List[str],
        target_id: str,
        operation: OperationType,
        columns_affected: Optional[List[str]] = None,
        transformation_logic: str = ""
    ) -> None:
        """
        Track a data transformation.
        
        Args:
            source_ids: IDs of source nodes
            target_id: ID of target node
            operation: Type of operation
            columns_affected: Columns involved in transformation
            transformation_logic: Description of transformation
        """
        for source_id in source_ids:
            edge = LineageEdge(
                source_id=source_id,
                target_id=target_id,
                operation=operation,
                columns_affected=columns_affected or [],
                transformation_logic=transformation_logic
            )
            self.edges.append(edge)
        
        if self._current_record:
            self._current_record.operations.append({
                "sources": source_ids,
                "target": target_id,
                "operation": operation.value,
                "columns": columns_affected,
                "logic": transformation_logic,
                "timestamp": datetime.now().isoformat()
            })
        
        self._save_lineage()
        logger.debug(f"Tracked transformation: {source_ids} -> {target_id}")
    
    def get_upstream(self, node_id: str) -> List[LineageNode]:
        """Get all upstream nodes for a given node."""
        upstream_ids = set()
        
        def _collect_upstream(nid: str):
            for edge in self.edges:
                if edge.target_id == nid:
                    upstream_ids.add(edge.source_id)
                    _collect_upstream(edge.source_id)
        
        _collect_upstream(node_id)
        
        return [self.nodes[uid] for uid in upstream_ids if uid in self.nodes]
    
    def get_downstream(self, node_id: str) -> List[LineageNode]:
        """Get all downstream nodes for a given node."""
        downstream_ids = set()
        
        def _collect_downstream(nid: str):
            for edge in self.edges:
                if edge.source_id == nid:
                    downstream_ids.add(edge.target_id)
                    _collect_downstream(edge.target_id)
        
        _collect_downstream(node_id)
        
        return [self.nodes[did] for did in downstream_ids if did in self.nodes]
    
    def get_column_lineage(self, target_column: str) -> Dict[str, Any]:
        """
        Get column-level lineage for a specific column.
        
        Args:
            target_column: Column to trace
            
        Returns:
            Column lineage information
        """
        lineage = {
            "column": target_column,
            "transformations": []
        }
        
        for edge in self.edges:
            if target_column in edge.columns_affected:
                lineage["transformations"].append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "operation": edge.operation.value,
                    "logic": edge.transformation_logic
                })
        
        return lineage
    
    def visualize_lineage(self, output_path: str = "reports/lineage") -> str:
        """
        Generate lineage visualization (Mermaid diagram).
        
        Args:
            output_path: Output directory
            
        Returns:
            Path to generated file
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Generate Mermaid diagram
        mermaid = "```mermaid\ngraph LR\n"
        
        # Add nodes
        for node_id, node in self.nodes.items():
            shape_start = "((" if node.node_type == "source" else "[["
            shape_end = "))" if node.node_type == "source" else "]]"
            mermaid += f"    {node_id}{shape_start}{node.name}{shape_end}\n"
        
        # Add edges
        for edge in self.edges:
            label = edge.operation.value
            mermaid += f"    {edge.source_id} -->|{label}| {edge.target_id}\n"
        
        mermaid += "```"
        
        # Write to file
        output_file = os.path.join(output_path, "lineage_diagram.md")
        with open(output_file, 'w') as f:
            f.write(f"# Data Lineage\n\n{mermaid}\n")
        
        logger.info(f"Generated lineage diagram: {output_file}")
        return output_file
    
    def export_lineage(self, format: str = "json") -> Dict[str, Any]:
        """
        Export complete lineage graph.
        
        Args:
            format: Export format (json)
            
        Returns:
            Lineage graph data
        """
        return {
            "nodes": [
                {"id": n.id, "name": n.name, "type": n.node_type, "metadata": n.metadata}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "operation": e.operation.value,
                    "columns": e.columns_affected
                }
                for e in self.edges
            ]
        }
