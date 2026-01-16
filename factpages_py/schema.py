"""
Schema Registry

Manages table metadata, column information, and relationships for knowledge graph
construction. Stores API-provided column aliases and enables auto-detection of
foreign key relationships based on column naming patterns.

Example:
    >>> from factpages_py import Factpages
    >>> fp = Factpages()
    >>>
    >>> # Get table configuration
    >>> config = fp.schema.get_table('wellbore')
    >>> print(config.id_field)  # 'wlbNpdidWellbore'
    >>> print(config.node_type)  # 'Wellbore'
    >>>
    >>> # Get connections from a table
    >>> connections = fp.schema.connections_from('wellbore')
    >>> for conn in connections:
    ...     print(f"{conn.source_table} --[{conn.connection_name}]--> {conn.target_table}")
    >>>
    >>> # Cache column metadata from API
    >>> fp.cache_schema('wellbore')  # Stores aliases, types, and detects ID columns
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Any

import pandas as pd

if TYPE_CHECKING:
    from .database import Database


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ColumnInfo:
    """Metadata for a single column, from API."""
    name: str
    alias: str
    type: str = "esriFieldTypeString"
    nullable: bool = True
    length: Optional[int] = None
    # Detected relationships
    references_table: Optional[str] = None  # If this column is a FK to another table
    is_primary_id: bool = False  # If this is the primary ID column for its table


@dataclass
class TableConfig:
    """Configuration for an entity table."""
    name: str
    table_id: int
    node_type: str
    id_field: str
    name_field: str
    has_geometry: bool = False
    rename: Dict[str, str] = field(default_factory=dict)
    foreign_keys: Dict[str, dict] = field(default_factory=dict)

    @property
    def id_field_renamed(self) -> str:
        """Get the renamed ID field (for export)."""
        return self.rename.get(self.id_field, self.id_field)

    @property
    def name_field_renamed(self) -> str:
        """Get the renamed name field (for export)."""
        return self.rename.get(self.name_field, self.name_field)

    def get_original_column(self, renamed: str) -> Optional[str]:
        """Get original column name from renamed column."""
        for orig, new in self.rename.items():
            if new == renamed:
                return orig
        return None

    def get_renamed_column(self, original: str) -> str:
        """Get renamed column name from original column."""
        return self.rename.get(original, original)


@dataclass
class ConnectionConfig:
    """Configuration for a connection between tables."""
    name: str  # Connection table name (e.g., 'field_licensee')
    table_id: int
    description: str
    source_table: str
    source_field: str
    target_table: str
    target_field: str
    connection_name: str  # Relationship type (e.g., 'OWNED_BY')
    properties: Dict[str, str] = field(default_factory=dict)

    @property
    def property_columns(self) -> List[str]:
        """Get list of property column names."""
        return list(self.properties.values())


@dataclass
class DirectConnection:
    """A direct foreign key connection within an entity table."""
    source_table: str
    source_field: str
    target_table: str
    target_field: str
    connection_name: str
    description: str


@dataclass
class ConnectionResult:
    """Result of extracting connections, ready for graph loading."""
    source_type: str
    target_type: str
    connection_name: str  # Also accessible as connection_type for rusty-graph compatibility
    source_id_field: str
    target_id_field: str
    data: pd.DataFrame
    properties: Dict[str, str] = field(default_factory=dict)

    @property
    def connection_type(self) -> str:
        """Alias for connection_name (rusty-graph uses connection_type)."""
        return self.connection_name

    def to_dict(self) -> dict:
        """Convert to dict format for rusty-graph."""
        return {
            'source_type': self.source_type,
            'target_type': self.target_type,
            'connection_type': self.connection_name,
            'source_id_field': self.source_id_field,
            'target_id_field': self.target_id_field,
            'data': self.data,
            'properties': self.properties
        }


# =============================================================================
# Schema Registry
# =============================================================================

class SchemaRegistry:
    """
    Registry for table metadata, column information, and relationships.

    Loads configuration from schema.json and provides methods to:
    - Get table configurations
    - Find connections between tables
    - Store and retrieve column metadata (aliases, types) from API
    - Auto-detect foreign keys based on column alias patterns
    - Apply column renames at export time
    """

    # Alias patterns that indicate foreign keys (alias -> target table)
    NPDID_ALIAS_PATTERNS = {
        'NPDID wellbore': 'wellbore',
        'NPDID field': 'field',
        'NPDID discovery': 'discovery',
        'NPDID facility': 'facility',
        'NPDID company': 'company',
        'NPDID licence': 'licence',
        'NPDID pipeline': 'pipeline',
        'NPDID drilling operator': 'company',
        'NPDID operating company': 'company',
        'NPDID responsible company': 'company',
    }

    def __init__(self, db: "Database", config_path: Optional[Path] = None):
        """
        Initialize the schema registry.

        Args:
            db: Database instance
            config_path: Path to schema.json (default: package directory)
        """
        self.db = db

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "schema.json"

        self._config_path = config_path
        self._columns_path = config_path.parent / "columns.json"
        self._config: dict = {}
        self._tables: Dict[str, TableConfig] = {}
        self._connections: Dict[str, ConnectionConfig] = {}
        self._id_patterns: Dict[str, List[str]] = {}
        # Connection names: "source:target" -> {"name": ..., "reverse": ...}
        self._connection_names: Dict[str, dict] = {}
        # Column metadata: dataset -> column_name -> ColumnInfo
        self._columns: Dict[str, Dict[str, ColumnInfo]] = {}
        self._load_config()
        self._load_columns()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self._config_path.exists():
            return

        with open(self._config_path) as f:
            self._config = json.load(f)

        # Parse entity tables
        for name, cfg in self._config.get('entities', {}).items():
            self._tables[name] = TableConfig(
                name=name,
                table_id=cfg.get('table_id', 0),
                node_type=cfg.get('node_type', name.title()),
                id_field=cfg.get('id_field', ''),
                name_field=cfg.get('name_field', ''),
                has_geometry=cfg.get('has_geometry', False),
                rename=cfg.get('rename', {}),
                foreign_keys=cfg.get('foreign_keys', {})
            )

        # Parse connection tables
        for name, cfg in self._config.get('connection_tables', {}).items():
            self._connections[name] = ConnectionConfig(
                name=name,
                table_id=cfg.get('table_id', 0),
                description=cfg.get('description', ''),
                source_table=cfg['source']['table'],
                source_field=cfg['source']['field'],
                target_table=cfg['target']['table'],
                target_field=cfg['target']['field'],
                connection_name=cfg.get('connection_name', 'RELATED_TO'),
                properties=cfg.get('properties', {})
            )

        # Parse ID patterns for auto-detection
        self._id_patterns = self._config.get('id_patterns', {})

        # Parse connection names (for auto-detected connections)
        self._connection_names = self._config.get('connection_names', {})

    def _load_columns(self) -> None:
        """Load column metadata from cache file."""
        if not self._columns_path.exists():
            return

        try:
            with open(self._columns_path) as f:
                data = json.load(f)
            # Reconstruct ColumnInfo objects
            for dataset, columns in data.items():
                self._columns[dataset] = {}
                for col_name, col_data in columns.items():
                    self._columns[dataset][col_name] = ColumnInfo(
                        name=col_data['name'],
                        alias=col_data['alias'],
                        type=col_data.get('type', 'esriFieldTypeString'),
                        nullable=col_data.get('nullable', True),
                        length=col_data.get('length'),
                        references_table=col_data.get('references_table'),
                        is_primary_id=col_data.get('is_primary_id', False)
                    )
        except (json.JSONDecodeError, IOError):
            self._columns = {}

    def _save_columns(self) -> None:
        """Save column metadata to cache file."""
        # Convert ColumnInfo objects to dicts for JSON serialization
        data = {}
        for dataset, columns in self._columns.items():
            data[dataset] = {}
            for col_name, col_info in columns.items():
                data[dataset][col_name] = {
                    'name': col_info.name,
                    'alias': col_info.alias,
                    'type': col_info.type,
                    'nullable': col_info.nullable,
                    'length': col_info.length,
                    'references_table': col_info.references_table,
                    'is_primary_id': col_info.is_primary_id
                }
        with open(self._columns_path, 'w') as f:
            json.dump(data, f, indent=2)

    def reload(self) -> None:
        """Reload configuration from disk (for dynamic updates)."""
        self._tables.clear()
        self._connections.clear()
        self._id_patterns.clear()
        self._connection_names.clear()
        self._columns.clear()
        self._load_config()
        self._load_columns()

    # =========================================================================
    # Table Access
    # =========================================================================

    def get_table(self, name: str) -> Optional[TableConfig]:
        """Get configuration for a table."""
        return self._tables.get(name)

    def list_tables(self) -> List[str]:
        """List all configured entity tables."""
        return list(self._tables.keys())

    def get_connection_table(self, name: str) -> Optional[ConnectionConfig]:
        """Get configuration for a connection table."""
        return self._connections.get(name)

    def list_connection_tables(self) -> List[str]:
        """List all configured connection tables."""
        return list(self._connections.keys())

    # =========================================================================
    # Connection Discovery
    # =========================================================================

    def connections_from(self, table: str) -> List[DirectConnection]:
        """
        Get all direct connections from a table (via foreign keys).

        Args:
            table: Source table name

        Returns:
            List of DirectConnection objects

        Example:
            >>> connections = fp.connectors.connections_from('wellbore')
            >>> for c in connections:
            ...     print(f"wellbore --[{c.connection_name}]--> {c.target_table}")
        """
        config = self._tables.get(table)
        if config is None:
            return []

        connections = []
        for fk_field, fk_info in config.foreign_keys.items():
            connections.append(DirectConnection(
                source_table=table,
                source_field=config.id_field,
                target_table=fk_info['target_table'],
                target_field=fk_info['target_field'],
                connection_name=fk_info.get('connection_name', 'RELATED_TO'),
                description=fk_info.get('description', '')
            ))

        return connections

    def connections_to(self, table: str) -> List[DirectConnection]:
        """
        Get all direct connections to a table (reverse foreign keys).

        Args:
            table: Target table name

        Returns:
            List of DirectConnection objects
        """
        connections = []

        for source_name, source_config in self._tables.items():
            for fk_field, fk_info in source_config.foreign_keys.items():
                if fk_info['target_table'] == table:
                    connections.append(DirectConnection(
                        source_table=source_name,
                        source_field=source_config.id_field,
                        target_table=table,
                        target_field=fk_info['target_field'],
                        connection_name=fk_info.get('connection_name', 'RELATED_TO'),
                        description=fk_info.get('description', '')
                    ))

        return connections

    def via_tables_from(self, table: str) -> List[ConnectionConfig]:
        """
        Get all connection tables that link FROM this table.

        Args:
            table: Source table name

        Returns:
            List of ConnectionConfig objects
        """
        return [
            conn for conn in self._connections.values()
            if conn.source_table == table
        ]

    def via_tables_to(self, table: str) -> List[ConnectionConfig]:
        """
        Get all connection tables that link TO this table.

        Args:
            table: Target table name

        Returns:
            List of ConnectionConfig objects
        """
        return [
            conn for conn in self._connections.values()
            if conn.target_table == table
        ]

    def all_connections_for(self, table: str) -> dict:
        """
        Get all connections for a table (both direct and via connection tables).

        Args:
            table: Table name

        Returns:
            Dict with 'direct_from', 'direct_to', 'via_from', 'via_to' keys
        """
        return {
            'direct_from': self.connections_from(table),
            'direct_to': self.connections_to(table),
            'via_from': self.via_tables_from(table),
            'via_to': self.via_tables_to(table)
        }

    # =========================================================================
    # Connection Extraction
    # =========================================================================

    def extract_direct_connections(
        self,
        source_table: str,
        target_table: str,
        rename: bool = False
    ) -> Optional[ConnectionResult]:
        """
        Extract direct connections between two entity tables.

        Uses foreign keys defined in the source table.

        Args:
            source_table: Source entity table
            target_table: Target entity table
            rename: Apply column renames to output

        Returns:
            ConnectionResult or None if no connection exists
        """
        source_config = self._tables.get(source_table)
        if source_config is None:
            return None

        # Find the foreign key that connects to target
        fk_info = None
        fk_field = None
        for field, info in source_config.foreign_keys.items():
            if info['target_table'] == target_table:
                fk_field = field
                fk_info = info
                break

        if fk_info is None:
            return None

        # Load source data
        df = self.db.get_or_none(source_table)
        if df is None or df.empty:
            return None

        # Extract connection pairs
        source_id = source_config.id_field
        target_id = fk_field

        if source_id not in df.columns or target_id not in df.columns:
            return None

        conn_df = df[[source_id, target_id]].dropna().drop_duplicates()
        conn_df[source_id] = conn_df[source_id].astype(int)
        conn_df[target_id] = conn_df[target_id].astype(int)

        # Rename to source_id, target_id for consistency
        conn_df = conn_df.rename(columns={
            source_id: 'source_id',
            target_id: 'target_id'
        })

        target_config = self._tables.get(target_table)

        return ConnectionResult(
            source_type=source_config.node_type,
            target_type=target_config.node_type if target_config else target_table.title(),
            connection_name=fk_info.get('connection_name', 'RELATED_TO'),
            source_id_field=source_config.id_field_renamed if rename else source_config.id_field,
            target_id_field=target_config.id_field_renamed if rename and target_config else fk_info['target_field'],
            data=conn_df.reset_index(drop=True)
        )

    def extract_via_connections(
        self,
        connection_table: str,
        rename: bool = False,
        include_properties: bool = True
    ) -> Optional[ConnectionResult]:
        """
        Extract connections from a connection table.

        Args:
            connection_table: Name of connection table
            rename: Apply column renames
            include_properties: Include property columns

        Returns:
            ConnectionResult or None
        """
        conn_config = self._connections.get(connection_table)
        if conn_config is None:
            return None

        df = self.db.get_or_none(connection_table)
        if df is None or df.empty:
            return None

        source_id = conn_config.source_field
        target_id = conn_config.target_field

        if source_id not in df.columns or target_id not in df.columns:
            return None

        # Build column list
        columns = [source_id, target_id]
        if include_properties:
            columns.extend([
                col for col in conn_config.property_columns
                if col in df.columns
            ])

        conn_df = df[columns].dropna(subset=[source_id, target_id]).copy()
        conn_df[source_id] = conn_df[source_id].astype(int)
        conn_df[target_id] = conn_df[target_id].astype(int)

        # Rename source/target
        conn_df = conn_df.rename(columns={
            source_id: 'source_id',
            target_id: 'target_id'
        })

        source_config = self._tables.get(conn_config.source_table)
        target_config = self._tables.get(conn_config.target_table)

        return ConnectionResult(
            source_type=source_config.node_type if source_config else conn_config.source_table.title(),
            target_type=target_config.node_type if target_config else conn_config.target_table.title(),
            connection_name=conn_config.connection_name,
            source_id_field=source_config.id_field_renamed if rename and source_config else conn_config.source_field,
            target_id_field=target_config.id_field_renamed if rename and target_config else conn_config.target_field,
            data=conn_df.drop_duplicates().reset_index(drop=True),
            properties=conn_config.properties
        )

    def extract_auto_detected_connections(
        self,
        source_table: str,
        target_table: str,
        rename: bool = False
    ) -> Optional[ConnectionResult]:
        """
        Extract connections based on auto-detected column names.

        Uses NPDID column patterns to find foreign keys not explicitly
        configured in schema.json.

        Args:
            source_table: Source entity table
            target_table: Target entity table
            rename: Apply column renames to output

        Returns:
            ConnectionResult or None if no connection exists
        """
        source_config = self._tables.get(source_table)
        if source_config is None:
            return None

        # Get auto-detected connections for this source
        auto_connections = self.get_connections_by_columns(source_table)

        # Find columns that connect to target
        target_columns = [
            col for col, tgt in auto_connections.items()
            if tgt == target_table
        ]

        if not target_columns:
            return None

        # Use the first matching column (could be extended to handle multiple)
        target_col = target_columns[0]

        # Skip if this is already a configured foreign key
        if source_config.foreign_keys and target_col in source_config.foreign_keys:
            return None

        # Load source data
        df = self.db.get_or_none(source_table)
        if df is None or df.empty:
            return None

        source_id = source_config.id_field
        if source_id not in df.columns or target_col not in df.columns:
            return None

        conn_df = df[[source_id, target_col]].dropna().drop_duplicates()
        conn_df[source_id] = conn_df[source_id].astype(int)
        conn_df[target_col] = conn_df[target_col].astype(int)

        # Rename to source_id, target_id for consistency
        conn_df = conn_df.rename(columns={
            source_id: 'source_id',
            target_col: 'target_id'
        })

        target_config = self._tables.get(target_table)

        return ConnectionResult(
            source_type=source_config.node_type,
            target_type=target_config.node_type if target_config else target_table.title(),
            connection_name=self._generate_connection_name(source_table, target_table, target_col),
            source_id_field=source_config.id_field_renamed if rename else source_config.id_field,
            target_id_field=target_config.id_field_renamed if rename and target_config else target_col,
            data=conn_df.reset_index(drop=True)
        )

    # =========================================================================
    # Bulk Connection Export
    # =========================================================================

    def get_all_connectors(
        self,
        loaded_node_types: Optional[Set[str]] = None,
        rename: bool = False
    ) -> List[ConnectionResult]:
        """
        Get all valid connectors for loaded node types.

        This is the main method for bulk graph loading. If loaded_node_types
        is provided, only returns connections where both endpoints are loaded.

        Args:
            loaded_node_types: Set of node types in the graph (e.g., {'Field', 'Wellbore'})
            rename: Apply column renames

        Returns:
            List of ConnectionResult objects ready for graph loading

        Example:
            >>> graph = rusty_graph.KnowledgeGraph()
            >>> graph.add_nodes(fp.graph.nodes('field'), node_type='Field', ...)
            >>> graph.add_nodes(fp.graph.nodes('wellbore'), node_type='Wellbore', ...)
            >>>
            >>> # Get only valid connectors
            >>> connectors = fp.connectors.get_all_connectors(
            ...     loaded_node_types={'Field', 'Wellbore'}
            ... )
            >>> for conn in connectors:
            ...     graph.add_connections(conn.data, conn.source_type, conn.target_type)
        """
        results = []

        # 1. Direct connections from entity tables
        for source_name in self._tables:
            for target_name in self._tables:
                if source_name == target_name:
                    continue

                conn = self.extract_direct_connections(source_name, target_name, rename=rename)
                if conn is not None and not conn.data.empty:
                    # Filter by loaded node types
                    if loaded_node_types:
                        if conn.source_type not in loaded_node_types:
                            continue
                        if conn.target_type not in loaded_node_types:
                            continue
                    results.append(conn)

        # 2. Connections via connection tables
        for conn_table in self._connections:
            conn = self.extract_via_connections(conn_table, rename=rename)
            if conn is not None and not conn.data.empty:
                if loaded_node_types:
                    if conn.source_type not in loaded_node_types:
                        continue
                    if conn.target_type not in loaded_node_types:
                        continue
                results.append(conn)

        # 3. Auto-detected connections from column names
        seen_pairs = {(r.source_type, r.target_type, r.connection_name) for r in results}
        for source_name in self._tables:
            for target_name in self._tables:
                if source_name == target_name:
                    continue

                conn = self.extract_auto_detected_connections(source_name, target_name, rename=rename)
                if conn is not None and not conn.data.empty:
                    # Skip if we already have this connection type
                    key = (conn.source_type, conn.target_type, conn.connection_name)
                    if key in seen_pairs:
                        continue

                    if loaded_node_types:
                        if conn.source_type not in loaded_node_types:
                            continue
                        if conn.target_type not in loaded_node_types:
                            continue

                    results.append(conn)
                    seen_pairs.add(key)

        return results

    def connectors_summary(self) -> pd.DataFrame:
        """
        Get a summary of all available connectors.

        Returns:
            DataFrame with source, target, connection_name, table columns
        """
        rows = []

        # Direct connections
        for table_name, config in self._tables.items():
            for fk_field, fk_info in config.foreign_keys.items():
                rows.append({
                    'source': config.node_type,
                    'target': self._tables.get(fk_info['target_table'], TableConfig(
                        name=fk_info['target_table'],
                        table_id=0,
                        node_type=fk_info['target_table'].title(),
                        id_field='',
                        name_field=''
                    )).node_type,
                    'connection_name': fk_info.get('connection_name', 'RELATED_TO'),
                    'via_table': None,
                    'type': 'direct'
                })

        # Via connection tables
        for conn_name, conn_config in self._connections.items():
            source_config = self._tables.get(conn_config.source_table)
            target_config = self._tables.get(conn_config.target_table)
            rows.append({
                'source': source_config.node_type if source_config else conn_config.source_table.title(),
                'target': target_config.node_type if target_config else conn_config.target_table.title(),
                'connection_name': conn_config.connection_name,
                'via_table': conn_name,
                'type': 'via_table'
            })

        return pd.DataFrame(rows)

    # =========================================================================
    # Column Rename Utilities
    # =========================================================================

    def apply_renames(
        self,
        df: pd.DataFrame,
        table: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Apply column renames to a DataFrame.

        This is used at export time - database always stores original names.

        Args:
            df: Input DataFrame
            table: Table name (to get rename mapping)
            inplace: Modify DataFrame in place

        Returns:
            DataFrame with renamed columns
        """
        config = self._tables.get(table)
        if config is None or not config.rename:
            return df

        if not inplace:
            df = df.copy()

        # Only rename columns that exist
        rename_map = {
            orig: new for orig, new in config.rename.items()
            if orig in df.columns
        }

        return df.rename(columns=rename_map)

    def reverse_renames(
        self,
        df: pd.DataFrame,
        table: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Reverse column renames (renamed -> original).

        Args:
            df: DataFrame with renamed columns
            table: Table name
            inplace: Modify in place

        Returns:
            DataFrame with original column names
        """
        config = self._tables.get(table)
        if config is None or not config.rename:
            return df

        if not inplace:
            df = df.copy()

        # Build reverse mapping
        reverse_map = {
            new: orig for orig, new in config.rename.items()
            if new in df.columns
        }

        return df.rename(columns=reverse_map)

    # =========================================================================
    # Auto-Detection
    # =========================================================================

    def detect_foreign_keys(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect foreign keys in a DataFrame using ID patterns.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict mapping column name to target table
        """
        detected = {}

        for col in df.columns:
            for table, patterns in self._id_patterns.items():
                if col in patterns:
                    detected[col] = table
                    break

        return detected

    def detect_connections_for_df(
        self,
        df: pd.DataFrame,
        source_table: str
    ) -> List[dict]:
        """
        Detect potential connections for a DataFrame.

        Args:
            df: DataFrame to analyze
            source_table: Name of the source table

        Returns:
            List of detected connections with column info
        """
        source_config = self._tables.get(source_table)
        if source_config is None:
            return []

        detected_fks = self.detect_foreign_keys(df)
        connections = []

        for col, target_table in detected_fks.items():
            if target_table == source_table:
                continue  # Skip self-references

            target_config = self._tables.get(target_table)
            connections.append({
                'source_table': source_table,
                'source_id': source_config.id_field,
                'target_table': target_table,
                'target_id': col,
                'target_config': target_config,
                'values_count': df[col].notna().sum()
            })

        return connections

    # =========================================================================
    # Column Metadata Management
    # =========================================================================

    def set_columns(
        self,
        dataset: str,
        fields: List[dict],
        table_config: Optional[TableConfig] = None
    ) -> None:
        """
        Set column metadata for a dataset from API field definitions.

        Automatically detects foreign keys based on alias patterns like
        "NPDID wellbore", "NPDID field", etc.

        Args:
            dataset: Dataset name
            fields: List of field definitions from API metadata
            table_config: Optional table config to mark primary ID

        Example:
            >>> metadata = fp.get_metadata('wellbore')
            >>> fp.schema.set_columns('wellbore', metadata['fields'])
        """
        self._columns[dataset] = {}
        primary_id = table_config.id_field if table_config else None

        for field in fields:
            name = field.get('name', '')
            alias = field.get('alias', name)

            # Detect if this column references another table
            references = self._detect_reference_from_alias(alias)

            col_info = ColumnInfo(
                name=name,
                alias=alias,
                type=field.get('type', 'esriFieldTypeString'),
                nullable=field.get('nullable', True),
                length=field.get('length'),
                references_table=references,
                is_primary_id=(name == primary_id)
            )
            self._columns[dataset][name] = col_info

        self._save_columns()

    def _detect_reference_from_alias(self, alias: str) -> Optional[str]:
        """
        Detect what table a column references based on its alias.

        Uses patterns like "NPDID wellbore" to identify foreign keys.
        """
        alias_lower = alias.lower()
        for pattern, table in self.NPDID_ALIAS_PATTERNS.items():
            if pattern.lower() in alias_lower:
                return table
        return None

    def get_column(self, column: str, dataset: Optional[str] = None) -> Optional[ColumnInfo]:
        """
        Get metadata for a column.

        Args:
            column: Column name
            dataset: Optional dataset to search in (searches all if not specified)

        Returns:
            ColumnInfo or None
        """
        if dataset:
            return self._columns.get(dataset, {}).get(column)

        # Search all datasets
        for columns in self._columns.values():
            if column in columns:
                return columns[column]
        return None

    def get_columns(self, dataset: str) -> Dict[str, ColumnInfo]:
        """Get all column metadata for a dataset."""
        return self._columns.get(dataset, {})

    def get_alias(self, column: str, dataset: Optional[str] = None) -> Optional[str]:
        """
        Get the readable alias for a column.

        Args:
            column: Column name
            dataset: Optional dataset to search in

        Returns:
            Readable alias or None if not found
        """
        col_info = self.get_column(column, dataset)
        return col_info.alias if col_info else None

    def get_all_aliases(self) -> Dict[str, str]:
        """
        Get all column aliases merged into a single dict.

        Since column names are unique per prefix (wlb*, fld*, dsc*, etc.),
        this is safe to use for general lookups.

        Returns:
            Dict mapping all column names to their aliases
        """
        merged = {}
        for columns in self._columns.values():
            for name, info in columns.items():
                merged[name] = info.alias
        return merged

    def has_columns(self, dataset: str) -> bool:
        """Check if column metadata is cached for a dataset."""
        return dataset in self._columns and bool(self._columns[dataset])

    def get_foreign_keys_by_alias(self, dataset: str) -> Dict[str, str]:
        """
        Get columns that reference other tables, detected via alias patterns.

        Returns:
            Dict mapping column name to target table name
        """
        columns = self._columns.get(dataset, {})
        return {
            name: info.references_table
            for name, info in columns.items()
            if info.references_table is not None
        }

    def get_id_columns(self, dataset: str) -> List[str]:
        """
        Get columns that are likely ID columns (reference other tables).

        Returns:
            List of column names that are foreign keys
        """
        return list(self.get_foreign_keys_by_alias(dataset).keys())

    @property
    def columns_path(self) -> Path:
        """Path to the column metadata cache file."""
        return self._columns_path

    # =========================================================================
    # Auto-Detection via Column Names
    # =========================================================================

    # Map column name prefixes to table names
    NPDID_PREFIX_MAP = {
        'wlb': 'wellbore',
        'fld': 'field',
        'dsc': 'discovery',
        'fcl': 'facility',
        'cmp': 'company',
        'prl': 'licence',
        'pip': 'pipeline',
        'ply': 'play',
        'baa': 'bsns_arr_area',
    }

    def parse_npdid_column(self, column: str) -> Optional[str]:
        """
        Parse an NPDID column name to determine what table it references.

        Column names follow patterns like:
        - wlbNpdidWellbore -> wellbore
        - fldNpdidField -> field
        - cmpNpdidCompany -> company

        Args:
            column: Column name

        Returns:
            Target table name or None if not an NPDID column
        """
        col_lower = column.lower()
        if 'npdid' not in col_lower:
            return None

        # Get the prefix (first 3 chars typically)
        prefix = column[:3].lower()
        return self.NPDID_PREFIX_MAP.get(prefix)

    def get_npdid_columns(self, dataset: str) -> Dict[str, str]:
        """
        Get all NPDID columns in a dataset and what they reference.

        Args:
            dataset: Dataset name

        Returns:
            Dict mapping column name to target table
        """
        columns = self._columns.get(dataset, {})
        result = {}

        for col_name in columns:
            target = self.parse_npdid_column(col_name)
            if target:
                result[col_name] = target

        return result

    def get_primary_id(self, dataset: str) -> Optional[str]:
        """
        Get the primary NPDID column for a dataset.

        The primary ID follows the pattern {prefix}Npdid{TableName}:
        - wlbNpdidWellbore for wellbore
        - fldNpdidField for field
        - dscNpdidDiscovery for discovery

        Args:
            dataset: Dataset name

        Returns:
            Primary ID column name or None
        """
        # Map dataset names to expected column suffixes
        dataset_to_suffix = {
            'wellbore': 'Wellbore',
            'field': 'Field',
            'discovery': 'Discovery',
            'facility': 'Facility',
            'company': 'Company',
            'licence': 'Licence',
            'pipeline': 'Pipeline',
            'play': 'Play',
            'bsns_arr_area': 'BsnsArrArea',
        }

        expected_suffix = dataset_to_suffix.get(dataset)
        if not expected_suffix:
            return None

        # Look for column matching {prefix}Npdid{Suffix}
        columns = self._columns.get(dataset, {})
        for col_name in columns:
            if 'npdid' in col_name.lower() and col_name.endswith(expected_suffix):
                return col_name

        return None

    def get_connections_by_columns(self, dataset: str) -> Dict[str, str]:
        """
        Get foreign key connections detected via column names.

        Finds NPDID columns that reference other tables (not self).

        Args:
            dataset: Dataset name

        Returns:
            Dict mapping column name to target table
        """
        npdid_cols = self.get_npdid_columns(dataset)
        primary_id = self.get_primary_id(dataset)

        # Filter out the primary key
        return {
            col: target
            for col, target in npdid_cols.items()
            if col != primary_id
        }

    def analyze_dataset(self, dataset: str) -> dict:
        """
        Analyze a dataset's structure including columns and connections.

        Args:
            dataset: Dataset name

        Returns:
            Dict with structure info
        """
        columns = self._columns.get(dataset, {})
        primary_id = self.get_primary_id(dataset)
        connections = self.get_connections_by_columns(dataset)

        return {
            'dataset': dataset,
            'total_columns': len(columns),
            'primary_id': primary_id,
            'connections': connections,
            'connection_count': len(connections),
        }

    def print_structure(self, datasets: Optional[List[str]] = None, show_incoming: bool = True) -> None:
        """
        Print the data structure showing entities and their connections.

        Args:
            datasets: List of datasets to analyze (uses cached if None)
            show_incoming: Also show incoming connections

        Example output:
            wellbore (126 columns)
              Primary ID: wlbNpdidWellbore

              Outgoing:
                -> field (DRILLED_ON via fldNpdidField)
                -> discovery (TESTED via dscNpdidDiscovery)

              Incoming:
                <- field (HAS_WELLS via wlbNpdidWellbore)
        """
        if datasets is None:
            datasets = list(self._columns.keys())

        if not datasets:
            print("No datasets cached. Run fp.cache_schema() first.")
            return

        print("\nData Structure")
        print("=" * 70)

        for dataset in sorted(datasets):
            info = self.analyze_dataset(dataset)
            outgoing = self.get_outgoing_connections(dataset)
            incoming = self.get_incoming_connections(dataset) if show_incoming else []

            # Header
            print(f"\n{dataset} ({info['total_columns']} columns)")

            # Primary ID
            if info['primary_id']:
                print(f"  Primary ID: {info['primary_id']}")
            else:
                print("  Primary ID: (none - connection table)")

            # Outgoing connections
            print("\n  Outgoing:")
            if outgoing:
                for conn in sorted(outgoing, key=lambda x: x['target']):
                    marker = ''
                    if conn['source_type'] == 'auto_detected':
                        marker = ' *'
                    elif conn['source_type'] == 'connection_table':
                        marker = ' [tbl]'
                    print(f"    -> {conn['target']} ({conn['connection_name']} via {conn['via_column']}){marker}")
            else:
                print("    (none)")

            # Incoming connections
            if show_incoming:
                print("\n  Incoming:")
                if incoming:
                    for conn in sorted(incoming, key=lambda x: x['source']):
                        marker = ''
                        if conn['source_type'] == 'auto_detected':
                            marker = ' *'
                        elif conn['source_type'] == 'connection_table':
                            marker = ' [tbl]'
                        print(f"    <- {conn['source']} ({conn['reverse_name']} via {conn['via_column']}){marker}")
                else:
                    print("    (none)")

        print("\n  Legend: * = auto-detected, [tbl] = via connection table")
        print()

    def get_structure_summary(self) -> pd.DataFrame:
        """
        Get a DataFrame summarizing the data structure.

        Returns:
            DataFrame with columns: source, target, via_column
        """
        rows = []

        for dataset in self._columns.keys():
            connections = self.get_connections_by_columns(dataset)
            for col, target in connections.items():
                rows.append({
                    'source': dataset,
                    'target': target,
                    'via_column': col
                })

        return pd.DataFrame(rows)

    # =========================================================================
    # Unified Connection System
    # =========================================================================

    def get_outgoing_connections(self, entity: str) -> List[dict]:
        """
        Get all outgoing connections FROM an entity (entity -> other).

        Combines:
        - Foreign keys defined in schema.json
        - Connection tables where entity is source
        - Auto-detected connections from column names

        Returns list of dicts with:
        - target: target entity name
        - via_column: column name used for connection
        - connection_name: name of the relationship (e.g., DRILLED_ON)
        - properties: dict of property columns (for connection tables)
        - source_type: 'foreign_key', 'connection_table', or 'auto_detected'
        """
        connections = []
        seen_targets = set()

        # 1. Configured foreign keys from schema.json
        config = self._tables.get(entity)
        if config and config.foreign_keys:
            for col, fk_info in config.foreign_keys.items():
                target = fk_info.get('target_table')
                connections.append({
                    'target': target,
                    'via_column': col,
                    'connection_name': fk_info.get('connection_name', 'RELATED_TO'),
                    'description': fk_info.get('description', ''),
                    'properties': {},
                    'source_type': 'foreign_key'
                })
                seen_targets.add((target, col))

        # 2. Connection tables where this entity is the source
        for name, conn_config in self._connections.items():
            if conn_config.source_table == entity:
                connections.append({
                    'target': conn_config.target_table,
                    'via_column': f"[{name}]",  # Via connection table
                    'connection_name': conn_config.connection_name,
                    'description': conn_config.description,
                    'properties': conn_config.properties,
                    'source_type': 'connection_table',
                    'connection_table': name
                })

        # 3. Auto-detected from column names (not already covered)
        auto_detected = self.get_connections_by_columns(entity)
        for col, target in auto_detected.items():
            if (target, col) not in seen_targets:
                connections.append({
                    'target': target,
                    'via_column': col,
                    'connection_name': self._generate_connection_name(entity, target, col),
                    'description': f'Auto-detected via {col}',
                    'properties': {},
                    'source_type': 'auto_detected'
                })

        return connections

    def get_incoming_connections(self, entity: str) -> List[dict]:
        """
        Get all incoming connections TO an entity (other -> entity).

        Returns list of dicts with:
        - source: source entity name
        - via_column: column name in source table
        - connection_name: name of the relationship
        - reverse_name: suggested name for reverse relationship
        - source_type: 'foreign_key', 'connection_table', or 'auto_detected'
        """
        connections = []
        seen_sources = set()

        # 1. Find entities with foreign keys pointing to this entity
        for source_entity, config in self._tables.items():
            if source_entity == entity:
                continue

            if config.foreign_keys:
                for col, fk_info in config.foreign_keys.items():
                    if fk_info.get('target_table') == entity:
                        conn_name = fk_info.get('connection_name', 'RELATED_TO')
                        connections.append({
                            'source': source_entity,
                            'via_column': col,
                            'connection_name': conn_name,
                            'reverse_name': self._generate_reverse_name(source_entity, entity, conn_name),
                            'description': fk_info.get('description', ''),
                            'source_type': 'foreign_key'
                        })
                        seen_sources.add((source_entity, col))

        # 2. Connection tables where this entity is the target
        for name, conn_config in self._connections.items():
            if conn_config.target_table == entity:
                connections.append({
                    'source': conn_config.source_table,
                    'via_column': f"[{name}]",
                    'connection_name': conn_config.connection_name,
                    'reverse_name': self._generate_reverse_name(
                        conn_config.source_table, entity, conn_config.connection_name
                    ),
                    'description': conn_config.description,
                    'properties': conn_config.properties,
                    'source_type': 'connection_table',
                    'connection_table': name
                })

        # 3. Auto-detected from column names
        for source_entity in self._columns.keys():
            if source_entity == entity:
                continue

            auto_detected = self.get_connections_by_columns(source_entity)
            for col, target in auto_detected.items():
                if target == entity and (source_entity, col) not in seen_sources:
                    conn_name = self._generate_connection_name(source_entity, entity, col)
                    connections.append({
                        'source': source_entity,
                        'via_column': col,
                        'connection_name': conn_name,
                        'reverse_name': self._generate_reverse_name(source_entity, entity, conn_name),
                        'description': f'Auto-detected via {col}',
                        'source_type': 'auto_detected'
                    })

        return connections

    def _generate_connection_name(self, source: str, target: str, column: str = None) -> str:
        """
        Generate a connection name based on source and target.

        Priority:
        1. Custom names from imported connections_config.json
        2. Names from schema.json connection_names section
        3. Generic auto-generated name
        """
        # 1. Check for custom config from import
        if column and hasattr(self, '_custom_connection_names'):
            key = (source, target, column)
            if key in self._custom_connection_names:
                return self._custom_connection_names[key]

        # 2. Check schema.json connection_names
        schema_key = f"{source}:{target}"
        if schema_key in self._connection_names:
            return self._connection_names[schema_key].get('name', f"RELATED_TO_{target.upper()}")

        # 3. Generic fallback
        return f"RELATED_TO_{target.upper()}"

    def _generate_reverse_name(self, source: str, target: str, forward_name: str = None) -> str:
        """
        Generate a reverse connection name.

        Priority:
        1. Custom names from imported connections_config.json
        2. Names from schema.json connection_names section
        3. Generic auto-generated name
        """
        # 1. Check for custom config from import (stored as reverse key)
        if hasattr(self, '_custom_connection_names'):
            # Check various key formats used during import
            for via_col in self._get_via_columns_for_pair(source, target):
                reverse_key = f"{(source, target, via_col)}_reverse"
                if reverse_key in self._custom_connection_names:
                    return self._custom_connection_names[reverse_key]

        # 2. Check schema.json connection_names
        schema_key = f"{source}:{target}"
        if schema_key in self._connection_names:
            return self._connection_names[schema_key].get('reverse', f"HAS_{source.upper()}S")

        # 3. Generic fallback
        return f"HAS_{source.upper()}S"

    def _get_via_columns_for_pair(self, source: str, target: str) -> List[str]:
        """Get all via_columns that connect source to target."""
        columns = []
        # Check auto-detected connections
        auto_detected = self.get_connections_by_columns(source)
        for col, tgt in auto_detected.items():
            if tgt == target:
                columns.append(col)
        # Check configured foreign keys
        config = self._tables.get(source)
        if config and config.foreign_keys:
            for col, fk_info in config.foreign_keys.items():
                if fk_info.get('target_table') == target:
                    columns.append(col)
        return columns

    def get_all_connections(self, entity: str) -> dict:
        """
        Get complete connection info for an entity.

        Returns dict with:
        - outgoing: list of outgoing connections
        - incoming: list of incoming connections
        """
        return {
            'outgoing': self.get_outgoing_connections(entity),
            'incoming': self.get_incoming_connections(entity)
        }

    def get_connections_dataframe(self) -> pd.DataFrame:
        """
        Get all connections as a DataFrame for graph building.

        Returns:
            DataFrame with columns: source, target, connection_name, via_column,
                                   reverse_name, source_type, has_properties
        """
        rows = []
        seen = set()

        all_entities = set(self._tables.keys()) | set(self._columns.keys())

        for entity in all_entities:
            for conn in self.get_outgoing_connections(entity):
                key = (entity, conn['target'], conn['via_column'])
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        'source': entity,
                        'target': conn['target'],
                        'connection_name': conn['connection_name'],
                        'via_column': conn['via_column'],
                        'reverse_name': self._generate_reverse_name(
                            entity, conn['target'], conn['connection_name']
                        ),
                        'source_type': conn['source_type'],
                        'has_properties': bool(conn.get('properties')),
                    })

        return pd.DataFrame(rows)

    # =========================================================================
    # Connections Config Export/Import
    # =========================================================================

    def export_connections_config(
        self,
        path: Optional[Path] = None,
        include_all_properties: bool = False
    ) -> dict:
        """
        Export the connections configuration for review and customization.

        Generates a JSON structure with all entities and their connections,
        including auto-generated connection names that can be customized.

        Args:
            path: Optional path to save the config (default: connections_config.json)
            include_all_properties: If True, include all columns as available properties

        Returns:
            Dict with the full connections configuration

        Example:
            >>> config = fp.schema.export_connections_config()
            >>> # Edit connections_config.json to customize names and properties
            >>> fp.schema.import_connections_config()
        """
        config = {
            "_comment": "Connection configuration for knowledge graph construction. Edit connection_name and properties as needed.",
            "_version": "1.0.0",
            "entities": {}
        }

        # Get all entities (from tables config and cached columns)
        all_entities = set(self._tables.keys()) | set(self._columns.keys())

        for entity in sorted(all_entities):
            entity_config = self._build_entity_config(entity, include_all_properties)
            if entity_config:
                config["entities"][entity] = entity_config

        # Save to file if path provided or use default
        if path is None:
            path = self._config_path.parent / "connections_config.json"

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        return config

    def _build_entity_config(self, entity: str, include_all_properties: bool = False) -> Optional[dict]:
        """Build configuration for a single entity."""
        primary_id = self.get_primary_id(entity)
        if not primary_id:
            # Skip non-entity tables (connection tables without primary ID)
            return None

        table_config = self._tables.get(entity)
        columns = self._columns.get(entity, {})

        # Get available property columns (non-ID columns)
        available_properties = []
        if include_all_properties and columns:
            available_properties = [
                col for col in columns.keys()
                if 'npdid' not in col.lower() and col != primary_id
            ]

        entity_cfg = {
            "primary_id": primary_id,
            "node_type": table_config.node_type if table_config else entity.title(),
            "column_count": len(columns),
            "outgoing": [],
            "incoming": []
        }

        # Build outgoing connections
        for conn in self.get_outgoing_connections(entity):
            out_cfg = {
                "target": conn["target"],
                "via_column": conn["via_column"],
                "connection_name": conn["connection_name"],
                "auto_detected": conn["source_type"] == "auto_detected",
                "source_properties": [],
                "target_properties": []
            }

            # Add available properties hint if requested
            if include_all_properties:
                out_cfg["_available_source_properties"] = available_properties[:10]  # Limit for readability
                target_cols = self._columns.get(conn["target"], {})
                target_props = [c for c in target_cols.keys() if 'npdid' not in c.lower()]
                out_cfg["_available_target_properties"] = target_props[:10]

            entity_cfg["outgoing"].append(out_cfg)

        # Build incoming connections
        for conn in self.get_incoming_connections(entity):
            in_cfg = {
                "source": conn["source"],
                "via_column": conn["via_column"],
                "connection_name": conn["connection_name"],
                "reverse_name": conn["reverse_name"],
                "auto_detected": conn["source_type"] == "auto_detected",
                "source_properties": [],
                "target_properties": []
            }

            if include_all_properties:
                source_cols = self._columns.get(conn["source"], {})
                source_props = [c for c in source_cols.keys() if 'npdid' not in c.lower()]
                in_cfg["_available_source_properties"] = source_props[:10]
                in_cfg["_available_target_properties"] = available_properties[:10]

            entity_cfg["incoming"].append(in_cfg)

        return entity_cfg

    def import_connections_config(
        self,
        path: Optional[Path] = None
    ) -> dict:
        """
        Import a connections configuration file.

        Loads customized connection names and property selections from
        the config file and applies them to the schema.

        Args:
            path: Path to config file (default: connections_config.json)

        Returns:
            Dict with import summary

        Example:
            >>> # After editing connections_config.json
            >>> summary = fp.schema.import_connections_config()
            >>> print(summary)
            {'connections_updated': 15, 'properties_configured': 8}
        """
        if path is None:
            path = self._config_path.parent / "connections_config.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Run export_connections_config() first to generate it."
            )

        with open(path) as f:
            config = json.load(f)

        # Store the imported config
        self._connections_config = config
        self._connections_config_path = path

        # Build lookup for connection names
        self._custom_connection_names = {}
        self._connection_properties = {}

        connections_updated = 0
        properties_configured = 0

        for entity, entity_cfg in config.get("entities", {}).items():
            # Process outgoing connections
            for conn in entity_cfg.get("outgoing", []):
                key = (entity, conn["target"], conn["via_column"])
                self._custom_connection_names[key] = conn["connection_name"]
                connections_updated += 1

                # Store property configuration
                if conn.get("source_properties") or conn.get("target_properties"):
                    self._connection_properties[key] = {
                        "source": conn.get("source_properties", []),
                        "target": conn.get("target_properties", [])
                    }
                    properties_configured += 1

            # Process incoming connections (for reverse names)
            for conn in entity_cfg.get("incoming", []):
                key = (conn["source"], entity, conn["via_column"])
                if "reverse_name" in conn:
                    # Store reverse name mapping
                    reverse_key = f"{key}_reverse"
                    self._custom_connection_names[reverse_key] = conn["reverse_name"]

        return {
            "connections_updated": connections_updated,
            "properties_configured": properties_configured,
            "config_path": str(path)
        }

    def get_connection_config(
        self,
        source: str,
        target: str,
        via_column: str
    ) -> Optional[dict]:
        """
        Get the configuration for a specific connection.

        Returns the connection name and property columns if configured.

        Args:
            source: Source entity
            target: Target entity
            via_column: Column used for the connection

        Returns:
            Dict with connection_name, source_properties, target_properties
        """
        key = (source, target, via_column)

        # Check for custom config
        if hasattr(self, '_custom_connection_names') and key in self._custom_connection_names:
            props = self._connection_properties.get(key, {})
            return {
                "connection_name": self._custom_connection_names[key],
                "source_properties": props.get("source", []),
                "target_properties": props.get("target", [])
            }

        # Fall back to auto-generated
        return {
            "connection_name": self._generate_connection_name(source, target),
            "source_properties": [],
            "target_properties": []
        }

    def get_configured_properties(
        self,
        source: str,
        target: str,
        via_column: str
    ) -> dict:
        """
        Get the property columns configured for a connection.

        Args:
            source: Source entity
            target: Target entity
            via_column: Column used for the connection

        Returns:
            Dict with source_properties and target_properties lists
        """
        key = (source, target, via_column)

        if hasattr(self, '_connection_properties') and key in self._connection_properties:
            return self._connection_properties[key]

        return {"source": [], "target": []}

    @property
    def connections_config_path(self) -> Path:
        """Path to the connections config file."""
        return self._config_path.parent / "connections_config.json"

    def has_connections_config(self) -> bool:
        """Check if a connections config file exists."""
        return self.connections_config_path.exists()
