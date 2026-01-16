"""
Graph Integration Helpers

Optimized endpoints for building knowledge graphs with rusty-graph.
Uses the SchemaRegistry for table metadata and relationship discovery.

Example (rusty-graph v0.3.24+):
    >>> from factpages_py import Factpages
    >>> import rusty_graph
    >>>
    >>> fp = Factpages()
    >>> graph = rusty_graph.KnowledgeGraph()
    >>>
    >>> # One-liner bulk loading
    >>> export = fp.graph.export_for_graph()
    >>> graph.add_nodes_bulk(export['nodes'])
    >>> graph.add_connections_from_source(export['connections'])
    >>>
    >>> # Or step by step with filtering
    >>> graph.add_nodes_bulk(fp.graph.all_node_specs(['field', 'wellbore']))
    >>> graph.add_connections_from_source(fp.graph.all_connection_specs())

Legacy example (manual iteration):
    >>> for table in ['field', 'discovery', 'wellbore']:
    ...     config = fp.schema.get_table(table)
    ...     df = fp.graph.nodes(table, rename=True)
    ...     graph.add_nodes(df, node_type=config.node_type,
    ...                     unique_id_field=config.id_field_renamed,
    ...                     node_title_field=config.name_field_renamed)
    >>>
    >>> for conn in fp.graph.all_connectors(graph.node_types):
    ...     graph.add_connections(conn.data, conn.source_type, conn.target_type,
    ...                          connection_type=conn.connection_type)
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Set

import pandas as pd

if TYPE_CHECKING:
    from .database import Database
    from .schema import SchemaRegistry, ConnectionResult, TableConfig


class GraphEndpoints:
    """
    Optimized endpoints for graph construction.

    Provides direct DataFrame access with minimal overhead,
    designed for bulk loading into knowledge graph libraries.

    Uses SchemaRegistry for table metadata and automatic
    connection discovery.
    """

    def __init__(self, db: "Database", schema: Optional["SchemaRegistry"] = None):
        self.db = db
        self._schema = schema

    @property
    def schema(self) -> "SchemaRegistry":
        """Lazy-load schema registry."""
        if self._schema is None:
            from .schema import SchemaRegistry
            self._schema = SchemaRegistry(self.db)
        return self._schema

    # Keep 'connectors' as alias for backwards compatibility
    @property
    def connectors(self) -> "SchemaRegistry":
        """Alias for schema (backwards compatibility)."""
        return self.schema

    # =========================================================================
    # Node Endpoints
    # =========================================================================

    def nodes(
        self,
        entity_type: str,
        rename: bool = False,
        include_geometry: bool = False
    ) -> pd.DataFrame:
        """
        Get nodes ready for graph loading.

        Args:
            entity_type: Entity type ('field', 'discovery', 'wellbore', etc.)
            rename: Apply column renames from connectors.json
            include_geometry: Include geometry column

        Returns:
            DataFrame ready for graph.add_nodes()

        Example:
            >>> df = fp.graph.nodes('wellbore', rename=True)
            >>> config = fp.connectors.get_table('wellbore')
            >>> graph.add_nodes(df, node_type=config.node_type,
            ...                 unique_id_field=config.id_field_renamed,
            ...                 node_title_field=config.name_field_renamed)
        """
        config = self.connectors.get_table(entity_type)

        df = self.db.get_or_none(entity_type)
        if df is None:
            return pd.DataFrame()

        df = df.copy()

        # Get ID field (from config or fallback)
        if config:
            id_field = config.id_field
        else:
            # Fallback to legacy mapping
            id_field = self._legacy_id_field(entity_type)

        # Ensure ID field is valid
        if id_field and id_field in df.columns:
            df = df.dropna(subset=[id_field])
            df[id_field] = df[id_field].astype(int)

        # Remove geometry if not needed
        if not include_geometry and '_geometry' in df.columns:
            df = df.drop(columns=['_geometry'])

        # Apply renames if requested
        if rename and config:
            df = self.connectors.apply_renames(df, entity_type)

        return df.reset_index(drop=True)

    def _legacy_id_field(self, entity_type: str) -> Optional[str]:
        """Fallback ID field mapping for unconfigured tables."""
        defaults = {
            'field': 'fldNpdidField',
            'discovery': 'dscNpdidDiscovery',
            'wellbore': 'wlbNpdidWellbore',
            'facility': 'fclNpdidFacility',
            'licence': 'prlNpdidLicence',
            'company': 'cmpNpdidCompany',
            'pipeline': 'pipNpdidPipeline',
            'play': 'plyNpdidPlay',
        }
        return defaults.get(entity_type)

    def all_nodes(
        self,
        rename: bool = False,
        include_geometry: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all entity types as DataFrames.

        Args:
            rename: Apply column renames
            include_geometry: Include geometry column

        Returns:
            Dict mapping entity type to DataFrame
        """
        result = {}
        for table_name in self.connectors.list_tables():
            df = self.nodes(table_name, rename=rename, include_geometry=include_geometry)
            if not df.empty:
                result[table_name] = df

        return result

    def node_info(self, entity_type: str) -> Optional["TableConfig"]:
        """
        Get table configuration for an entity type.

        Returns:
            TableConfig with id_field, name_field, node_type, etc.
        """
        return self.connectors.get_table(entity_type)

    # =========================================================================
    # Connection Endpoints
    # =========================================================================

    def connections(
        self,
        source_type: str,
        target_type: str,
        rename: bool = False
    ) -> pd.DataFrame:
        """
        Get connections between two entity types.

        Args:
            source_type: Source entity type
            target_type: Target entity type
            rename: Apply column renames

        Returns:
            DataFrame with source_id, target_id columns
        """
        conn = self.connectors.extract_direct_connections(
            source_type, target_type, rename=rename
        )
        if conn is not None:
            return conn.data
        return pd.DataFrame()

    def via_connections(
        self,
        connection_table: str,
        rename: bool = False,
        include_properties: bool = True
    ) -> Optional["ConnectionResult"]:
        """
        Get connections from a connection table (e.g., field_licensee).

        Args:
            connection_table: Name of connection table
            rename: Apply column renames
            include_properties: Include property columns

        Returns:
            ConnectionResult with data and metadata
        """
        return self.connectors.extract_via_connections(
            connection_table,
            rename=rename,
            include_properties=include_properties
        )

    def all_connectors(
        self,
        loaded_node_types: Optional[Set[str]] = None,
        rename: bool = False
    ) -> List["ConnectionResult"]:
        """
        Get all valid connectors for the loaded node types.

        This is the main method for bulk connection loading. Returns only
        connections where both endpoints are in loaded_node_types (if provided).

        Args:
            loaded_node_types: Set of node types in the graph
            rename: Apply column renames

        Returns:
            List of ConnectionResult objects

        Example:
            >>> # After loading nodes
            >>> connectors = fp.graph.all_connectors({'Field', 'Wellbore', 'Discovery'})
            >>> for conn in connectors:
            ...     graph.add_connections(conn.data, conn.source_type, conn.target_type)
        """
        return self.connectors.get_all_connectors(
            loaded_node_types=loaded_node_types,
            rename=rename
        )

    def connections_from(self, table: str) -> List[dict]:
        """
        Get all available connections from a table.

        Useful for exploring what connections are available.

        Args:
            table: Source table name

        Returns:
            List of connection info dicts

        Example:
            >>> for conn in fp.graph.connections_from('field'):
            ...     print(f"field --[{conn['connection_name']}]--> {conn['target']}")
        """
        result = []

        # Direct connections
        for dc in self.connectors.connections_from(table):
            result.append({
                'target': dc.target_table,
                'connection_name': dc.connection_name,
                'type': 'direct',
                'via_table': None,
                'description': dc.description
            })

        # Via connection tables
        for cc in self.connectors.via_tables_from(table):
            result.append({
                'target': cc.target_table,
                'connection_name': cc.connection_name,
                'type': 'via_table',
                'via_table': cc.name,
                'description': cc.description
            })

        return result

    def connections_summary(self) -> pd.DataFrame:
        """
        Get a summary of all available connections.

        Returns:
            DataFrame with source, target, connection_name columns
        """
        return self.connectors.connectors_summary()

    # =========================================================================
    # Geometry Endpoints
    # =========================================================================

    def geometries(
        self,
        entity_type: str,
        rename: bool = False
    ) -> pd.DataFrame:
        """
        Get geometries for an entity type.

        Args:
            entity_type: Entity type
            rename: If True, use renamed column names for id/name

        Returns:
            DataFrame with id, name, geometry columns
        """
        config = self.connectors.get_table(entity_type)
        if config is None:
            return pd.DataFrame()

        df = self.db.get_or_none(entity_type)
        if df is None or '_geometry' not in df.columns:
            return pd.DataFrame()

        id_field = config.id_field
        name_field = config.name_field

        result = df[[id_field, name_field, '_geometry']].copy()
        result = result.dropna(subset=[id_field, '_geometry'])

        # Rename to standard or renamed column names
        if rename:
            result = result.rename(columns={
                id_field: config.id_field_renamed,
                name_field: config.name_field_renamed,
                '_geometry': 'geometry'
            })
        else:
            result = result.rename(columns={
                id_field: 'id',
                name_field: 'name',
                '_geometry': 'geometry'
            })

        result[result.columns[0]] = result[result.columns[0]].astype(int)

        return result.reset_index(drop=True)

    # =========================================================================
    # Bulk Export (rusty-graph v0.3.24+ compatible)
    # =========================================================================

    def all_node_specs(
        self,
        tables: Optional[List[str]] = None,
        rename: bool = False,
        include_geometry: bool = False
    ) -> List[dict]:
        """
        Get all nodes as NodeSpec-compatible dicts for rusty-graph bulk loading.

        Compatible with rusty-graph v0.3.24+ add_nodes_bulk() method.

        Args:
            tables: Specific tables to include (all if None)
            rename: Apply column renames
            include_geometry: Include geometry data

        Returns:
            List of NodeSpec dicts ready for graph.add_nodes_bulk()

        Example:
            >>> specs = fp.graph.all_node_specs()
            >>> graph.add_nodes_bulk(specs)
        """
        if tables is None:
            tables = self.connectors.list_tables()

        specs = []
        for table_name in tables:
            config = self.connectors.get_table(table_name)
            if config is None:
                continue

            df = self.nodes(table_name, rename=rename, include_geometry=include_geometry)
            if df.empty:
                continue

            specs.append({
                'node_type': config.node_type,
                'unique_id_field': config.id_field_renamed if rename else config.id_field,
                'node_title_field': config.name_field_renamed if rename else config.name_field,
                'data': df
            })

        return specs

    def all_connection_specs(
        self,
        loaded_node_types: Optional[Set[str]] = None,
        rename: bool = False
    ) -> List[dict]:
        """
        Get all connections as ConnectionSpec-compatible dicts for rusty-graph.

        Compatible with rusty-graph v0.3.24+ add_connections_bulk() and
        add_connections_from_source() methods.

        Args:
            loaded_node_types: Filter to connections where both endpoints loaded
            rename: Apply column renames

        Returns:
            List of ConnectionSpec dicts ready for graph.add_connections_bulk()

        Example:
            >>> # Load all connections, let rusty-graph filter
            >>> specs = fp.graph.all_connection_specs()
            >>> graph.add_connections_from_source(specs)

            >>> # Or pre-filter to loaded types
            >>> specs = fp.graph.all_connection_specs(loaded_node_types={'Field', 'Wellbore'})
            >>> graph.add_connections_bulk(specs)
        """
        connectors = self.all_connectors(
            loaded_node_types=loaded_node_types,
            rename=rename
        )

        return [
            {
                'source_type': conn.source_type,
                'target_type': conn.target_type,
                'connection_name': conn.connection_name,
                'data': conn.data,
                'properties': conn.properties
            }
            for conn in connectors
        ]

    def export_for_graph(
        self,
        tables: Optional[List[str]] = None,
        rename: bool = False,
        include_geometry: bool = False
    ) -> dict:
        """
        Export all data for rusty-graph bulk loading.

        Compatible with rusty-graph v0.3.24+ bulk loading API.

        Args:
            tables: Specific tables to include (all if None)
            rename: Apply column renames
            include_geometry: Include geometry data

        Returns:
            Dict with 'nodes' (List[NodeSpec]) and 'connections' (List[ConnectionSpec])

        Example:
            >>> export = fp.graph.export_for_graph()
            >>> graph.add_nodes_bulk(export['nodes'])
            >>> graph.add_connections_from_source(export['connections'])
        """
        node_specs = self.all_node_specs(
            tables=tables,
            rename=rename,
            include_geometry=include_geometry
        )

        # Get loaded node types for connection filtering
        loaded_types = {spec['node_type'] for spec in node_specs}

        connection_specs = self.all_connection_specs(
            loaded_node_types=loaded_types,
            rename=rename
        )

        return {
            'nodes': node_specs,
            'connections': connection_specs
        }


# =============================================================================
# Helper Functions
# =============================================================================

def prepare_for_graph(
    df: pd.DataFrame,
    id_field: str,
    title_field: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare a DataFrame for rusty-graph.add_nodes().

    - Ensures ID field is present and unique
    - Drops rows with null IDs
    - Resets index
    - Handles NaN values

    Args:
        df: Input DataFrame
        id_field: Unique identifier field
        title_field: Optional display name field

    Returns:
        Cleaned DataFrame ready for graph loading
    """
    df = df.copy()
    df = df.dropna(subset=[id_field])
    df[id_field] = df[id_field].astype(int)
    return df.reset_index(drop=True)


def build_connection_df(
    source_df: pd.DataFrame,
    source_id: str,
    target_id: str
) -> pd.DataFrame:
    """
    Build a connection DataFrame from a source with foreign key.

    - Filters to non-null target IDs
    - Converts to int where needed
    - Resets index

    Args:
        source_df: DataFrame containing the relationship
        source_id: Source entity ID column
        target_id: Target entity ID column (foreign key)

    Returns:
        DataFrame with source_id, target_id columns
    """
    conn_df = source_df[source_df[target_id].notna()][[source_id, target_id]].copy()
    conn_df[source_id] = conn_df[source_id].astype(int)
    conn_df[target_id] = conn_df[target_id].astype(int)
    return conn_df.drop_duplicates().reset_index(drop=True)
