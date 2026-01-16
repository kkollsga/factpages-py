"""
Factpages API Client

Main entry point for the factpages_py library.
Provides both user-friendly entity access and efficient raw data access.

Example (user-friendly):
    >>> fp = Factpages()
    >>> troll = fp.field("troll")
    >>> print(troll.partners)
    >>> print(troll.production(2025, 8))

Example (raw data access):
    >>> fp = Factpages()
    >>> df = fp.download('discovery')
    >>> fields = fp.db.get('field')

Example (graph building):
    >>> fp = Factpages()
    >>> nodes = fp.graph.nodes('field')
    >>> connections = fp.graph.connections('discovery', 'field')
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from .datasets import LAYERS, TABLES, FACTMAPS_LAYERS
from .database import Database
from .entities import Field, Discovery, Wellbore, Company, License
from .entity_config import CustomEntity
from .graph import GraphEndpoints
from .supplementary import SupplementaryData
from .schema import SchemaRegistry


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ClientConfig:
    """
    Configuration for the Factpages API client.

    Provides fine-grained control over timeouts, retries, connection pooling,
    and caching behavior.

    Example:
        >>> config = ClientConfig(
        ...     timeout=60,
        ...     max_retries=5,
        ...     pool_connections=20
        ... )
        >>> fp = Factpages(config=config)
    """
    # Timeout settings
    timeout: int = 30
    connect_timeout: int = 10

    # Rate limiting
    rate_limit: float = 0.2  # Minimum seconds between requests

    # Retry settings
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    retry_status_forcelist: tuple = field(default_factory=lambda: (429, 500, 502, 503, 504))

    # Connection pooling
    pool_connections: int = 10  # Number of connection pools to cache
    pool_maxsize: int = 10  # Max connections per pool
    pool_block: bool = False  # Block when pool is full

    # Request caching
    enable_request_cache: bool = True
    cache_ttl: float = 5.0  # Seconds to cache in-flight request results


# =============================================================================
# In-Flight Request Cache
# =============================================================================

class RequestCache:
    """
    Cache for avoiding duplicate in-flight requests.

    When multiple threads request the same URL simultaneously,
    only one request is made and the result is shared.
    """

    def __init__(self, ttl: float = 5.0):
        self.ttl = ttl
        self._cache: dict[str, tuple[float, any]] = {}
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def _cache_key(self, url: str, params: dict) -> str:
        """Generate cache key from URL and params."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{url}:{param_str}".encode()).hexdigest()

    def get(self, url: str, params: dict) -> tuple[bool, any]:
        """
        Get cached response if available and fresh.

        Returns:
            Tuple of (found, value)
        """
        key = self._cache_key(url, params)

        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return True, value
                # Expired, remove it
                del self._cache[key]

        return False, None

    def set(self, url: str, params: dict, value: any) -> None:
        """Cache a response."""
        key = self._cache_key(url, params)
        with self._lock:
            self._cache[key] = (time.time(), value)

    def wait_or_start(self, url: str, params: dict) -> tuple[bool, any]:
        """
        Check if another thread is fetching this URL.

        Returns:
            Tuple of (should_fetch, cached_value)
            - If should_fetch is True, caller should make the request
            - If should_fetch is False, cached_value contains the result
        """
        key = self._cache_key(url, params)

        with self._lock:
            # Check cache first
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return False, value

            # Check if another thread is fetching
            if key in self._pending:
                event = self._pending[key]
            else:
                # We'll be the one to fetch
                self._pending[key] = threading.Event()
                return True, None

        # Wait for the other thread
        event.wait(timeout=30)

        # Get the result
        found, value = self.get(url, params)
        return False, value

    def complete(self, url: str, params: dict, value: any) -> None:
        """Mark a fetch as complete and notify waiters."""
        key = self._cache_key(url, params)

        with self._lock:
            self._cache[key] = (time.time(), value)
            if key in self._pending:
                self._pending[key].set()
                del self._pending[key]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class Factpages:
    """
    Main client for Norwegian Petroleum Factpages data.

    Provides three levels of access:
    1. Entity API - User-friendly object-oriented access
       >>> troll = fp.field("troll")
       >>> print(troll.production(2025, 8))

    2. Raw Data API - Direct DataFrame access
       >>> df = fp.download('discovery')
       >>> df = fp.db.get('field')

    3. Graph API - Optimized for knowledge graph building
       >>> nodes = fp.graph.nodes('field')
       >>> connections = fp.graph.connections('discovery', 'field')
    """

    # API Base URLs
    DATASERVICE_URL = "https://factmaps.sodir.no/api/rest/services/DataService/Data/FeatureServer"
    FACTMAPS_URL = "https://factmaps.sodir.no/api/rest/services/Factmaps/FactMapsWGS84/FeatureServer"

    def __init__(
        self,
        data_dir: Union[str, Path] = "./factpages_data",
        timeout: Optional[int] = None,
        rate_limit: Optional[float] = None,
        auto_sync: bool = False,
        config: Optional[ClientConfig] = None
    ):
        """
        Initialize the Factpages client.

        Args:
            data_dir: Directory for local database storage
            timeout: API request timeout in seconds (overrides config)
            rate_limit: Minimum seconds between API requests (overrides config)
            auto_sync: If True, auto-sync missing datasets on access
            config: Full client configuration (ClientConfig instance)

        Example:
            >>> # Simple usage with defaults
            >>> fp = Factpages()
            >>>
            >>> # Custom configuration
            >>> config = ClientConfig(
            ...     timeout=60,
            ...     max_retries=5,
            ...     pool_connections=20
            ... )
            >>> fp = Factpages(config=config)
        """
        # Use config or create default
        self.config = config or ClientConfig()

        # Allow individual overrides for backwards compatibility
        if timeout is not None:
            self.config.timeout = timeout
        if rate_limit is not None:
            self.config.rate_limit = rate_limit

        self.auto_sync = auto_sync

        # Create HTTP session with connection pooling and retry
        self._session = self._create_session()
        self._last_request_time = 0.0
        self._metadata_cache: dict[str, dict] = {}

        # In-flight request cache
        self._request_cache: Optional[RequestCache] = None
        if self.config.enable_request_cache:
            self._request_cache = RequestCache(ttl=self.config.cache_ttl)

        # Local database
        self.db = Database(data_dir)

        # Schema registry for table metadata
        self._schema = SchemaRegistry(self.db)

        # Graph endpoints for efficient data loading
        self.graph = GraphEndpoints(self.db, schema=self._schema)

        # Supplementary data manager
        self.supplementary = SupplementaryData(self.db)

    def _create_session(self) -> requests.Session:
        """
        Create an HTTP session with connection pooling and retry logic.

        Returns:
            Configured requests.Session
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'factpages-py/0.1.0'
        })

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=self.config.retry_status_forcelist,
            allowed_methods=["GET"],  # Only retry GET requests
        )

        # Configure connection pooling with retry
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            pool_block=self.config.pool_block,
            max_retries=retry_strategy
        )

        # Mount adapter for both HTTP and HTTPS
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self.config.timeout

    @property
    def rate_limit(self) -> float:
        """Minimum seconds between requests."""
        return self.config.rate_limit

    @property
    def schema(self) -> SchemaRegistry:
        """
        Schema registry for table metadata, column info, and relationships.

        Provides access to table configurations, column metadata (aliases, types),
        foreign key detection, and relationship mappings for knowledge graph construction.

        Example:
            >>> # Get table configuration
            >>> config = fp.schema.get_table('wellbore')
            >>> print(config.node_type)  # 'Wellbore'
            >>> print(config.id_field)   # 'wlbNpdidWellbore'
            >>>
            >>> # Get column alias
            >>> alias = fp.schema.get_alias('wlbWellboreName')
            >>> print(alias)  # 'Wellbore name'
            >>>
            >>> # Get detected foreign keys for a table
            >>> fks = fp.schema.get_foreign_keys_by_alias('field_licensee')
            >>> print(fks)  # {'fldNpdidField': 'field', 'cmpNpdidCompany': 'company'}
            >>>
            >>> # Reload config after editing schema.json
            >>> fp.schema.reload()
        """
        return self._schema

    # =========================================================================
    # Entity Access (User-Friendly API)
    # =========================================================================

    def field(self, name: str) -> Field:
        """
        Get a Field entity by name.

        Args:
            name: Field name (case-insensitive)

        Returns:
            Field object with properties and methods

        Example:
            >>> troll = fp.field("troll")
            >>> print(troll.operator)
            >>> print(troll.production(2025, 8))
        """
        fields = self._ensure_dataset('field')
        if fields is None:
            raise ValueError("Field data not available. Run sync() first.")

        # Case-insensitive search
        name_upper = name.upper()
        match = fields[fields['fldName'].str.upper() == name_upper]

        if match.empty:
            # Try partial match
            match = fields[fields['fldName'].str.upper().str.contains(name_upper, na=False)]

        if match.empty:
            raise ValueError(f"Field '{name}' not found.")

        return Field(match.iloc[0], self.db)

    def discovery(self, name: str) -> Discovery:
        """
        Get a Discovery entity by name.

        Args:
            name: Discovery name (case-insensitive)

        Returns:
            Discovery object
        """
        discoveries = self._ensure_dataset('discovery')
        if discoveries is None:
            raise ValueError("Discovery data not available. Run sync() first.")

        name_upper = name.upper()
        match = discoveries[discoveries['dscName'].str.upper() == name_upper]

        if match.empty:
            match = discoveries[discoveries['dscName'].str.upper().str.contains(name_upper, na=False)]

        if match.empty:
            raise ValueError(f"Discovery '{name}' not found.")

        return Discovery(match.iloc[0], self.db)

    def well(self, name: str) -> Wellbore:
        """
        Get a Wellbore entity by name.

        Args:
            name: Wellbore name (e.g., '35/11-25')

        Returns:
            Wellbore object
        """
        wellbores = self._ensure_dataset('wellbore')
        if wellbores is None:
            raise ValueError("Wellbore data not available. Run sync() first.")

        match = wellbores[wellbores['wlbWellboreName'] == name]

        if match.empty:
            # Try partial match
            match = wellbores[wellbores['wlbWellboreName'].str.contains(name, case=False, na=False)]

        if match.empty:
            raise ValueError(f"Wellbore '{name}' not found.")

        return Wellbore(match.iloc[0], self.db)

    def company(self, name: str) -> Company:
        """
        Get a Company entity by name.

        Args:
            name: Company name (partial match)

        Returns:
            Company object
        """
        companies = self._ensure_dataset('company')
        if companies is None:
            raise ValueError("Company data not available. Run sync() first.")

        match = companies[companies['cmpLongName'].str.contains(name, case=False, na=False)]

        if match.empty:
            raise ValueError(f"Company matching '{name}' not found.")

        return Company(match.iloc[0], self.db)

    def license(self, name: str) -> License:
        """
        Get a License entity by name.

        Args:
            name: License name (e.g., 'PL001')

        Returns:
            License object
        """
        licences = self._ensure_dataset('licence')
        if licences is None:
            raise ValueError("License data not available. Run sync() first.")

        match = licences[licences['prlName'] == name]

        if match.empty:
            match = licences[licences['prlName'].str.contains(name, case=False, na=False)]

        if match.empty:
            raise ValueError(f"License '{name}' not found.")

        return License(match.iloc[0], self.db)

    def custom_entity(self, entity_type: str, name: str) -> CustomEntity:
        """
        Get a custom entity by type and name.

        Custom entities are defined via JSON configuration files stored
        alongside sideloaded data. This allows flexible entity definitions
        without hardcoding them in the library.

        Args:
            entity_type: Entity type (e.g., "project", "asset")
            name: Entity name (case-insensitive)

        Returns:
            CustomEntity instance

        Example:
            >>> # Load data and register entity config
            >>> fp.load_json('projects.json', entity_config='project.entity.json')
            >>>
            >>> # Access custom entity
            >>> project = fp.custom_entity('project', 'Alpha')
            >>> print(project)
        """
        return self.supplementary.entity(entity_type, name)

    def load_json(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        key: Optional[str] = None,
        entity_config: Optional[Union[dict, str, Path]] = None,
        recursive: bool = True,
    ) -> dict:
        """
        Load external JSON data into the database.

        Sideloaded data is stored separately from API data (prefixed with
        'sideload_') to prevent data contamination. The source file path
        is tracked in metadata for easy identification and cleanup.

        Args:
            path: Path to JSON file
            name: Name for the dataset (defaults to filename)
            key: Specific key to extract from JSON (optional)
            entity_config: Entity configuration (dict or path to JSON file).
                          If provided, the entity type will be registered
                          and accessible via custom_entity().
            recursive: If True, recursively load nested tables

        Returns:
            Dict mapping dataset names to DataFrames

        Example:
            >>> # Simple load
            >>> fp.load_json('data.json')
            >>>
            >>> # Load with entity config for easy access
            >>> fp.load_json('projects.json', entity_config={
            ...     "entity_type": "Project",
            ...     "dataset": "sideload_projects",
            ...     "id_field": "id",
            ...     "name_field": "name",
            ...     "properties": {},
            ...     "related": {},
            ...     "display": {"title": "{name}"}
            ... })
            >>> project = fp.custom_entity('project', 'Alpha')
        """
        return self.supplementary.load_json(
            path=path,
            name=name,
            key=key,
            entity_config=entity_config,
            source=str(Path(path).resolve()),  # Track full path
            recursive=recursive,
        )

    def register_entity(
        self,
        config: Union[dict, str, Path],
        save: bool = True
    ):
        """
        Register a custom entity configuration.

        Once registered, the entity type can be accessed via custom_entity()
        and will persist across sessions.

        Args:
            config: Entity configuration dict or path to JSON file
            save: If True, save config to database (persists across sessions)

        Returns:
            The registered EntityConfig

        Example:
            >>> fp.register_entity({
            ...     "entity_type": "Asset",
            ...     "dataset": "sideload_assets",
            ...     "id_field": "asset_id",
            ...     "name_field": "asset_name",
            ...     "properties": {"status": {"column": "status"}},
            ...     "related": {},
            ...     "display": {"title": "{name}"}
            ... })
        """
        return self.supplementary.register_entity(config, save=save)

    def list_custom_entities(self) -> list:
        """
        List all registered custom entity types.

        Returns:
            List of entity type names
        """
        return self.supplementary.list_entities()

    def _ensure_dataset(self, dataset: str) -> Optional[pd.DataFrame]:
        """Ensure a dataset is available, auto-syncing if configured."""
        if self.db.has_dataset(dataset):
            return self.db.get(dataset)

        if self.auto_sync:
            print(f"Auto-syncing {dataset}...")
            df = self.download(dataset, progress=False)
            self.db.put(dataset, df, source='api')
            return df

        return None

    # =========================================================================
    # Raw Data API
    # =========================================================================

    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_timeout(self) -> tuple[int, int]:
        """Get timeout tuple (connect_timeout, read_timeout)."""
        return (self.config.connect_timeout, self.config.timeout)

    def _cached_get(self, url: str, params: dict) -> requests.Response:
        """
        Make a GET request with optional in-flight caching.

        This prevents duplicate concurrent requests to the same URL.
        """
        if self._request_cache is None:
            # No caching, just make the request
            self._rate_limit_wait()
            return self._session.get(url, params=params, timeout=self._get_timeout())

        # Check if we should fetch or wait for another thread
        should_fetch, cached = self._request_cache.wait_or_start(url, params)

        if not should_fetch:
            # Return cached response (as a mock response object)
            return cached

        # We need to fetch
        try:
            self._rate_limit_wait()
            response = self._session.get(url, params=params, timeout=self._get_timeout())
            response.raise_for_status()
            self._request_cache.complete(url, params, response)
            return response
        except Exception as e:
            # Clear the pending state so others don't wait forever
            self._request_cache.complete(url, params, None)
            raise

    def _get_url_and_id(self, dataset: str) -> tuple:
        """Get base URL and layer/table ID for a dataset."""
        if dataset in LAYERS:
            return self.DATASERVICE_URL, LAYERS[dataset]
        elif dataset in TABLES:
            return self.DATASERVICE_URL, TABLES[dataset]
        elif dataset in FACTMAPS_LAYERS:
            return self.FACTMAPS_URL, FACTMAPS_LAYERS[dataset]
        else:
            raise ValueError(
                f"Unknown dataset: '{dataset}'. "
                f"Use list_datasets() to see available options."
            )

    def list_datasets(self, category: str = "all") -> dict[str, int]:
        """
        List available API datasets.

        Args:
            category: 'layers', 'tables', 'factmaps', or 'all'

        Returns:
            Dict mapping dataset names to IDs
        """
        if category == "layers":
            return LAYERS.copy()
        elif category == "tables":
            return TABLES.copy()
        elif category == "factmaps":
            return FACTMAPS_LAYERS.copy()
        elif category == "all":
            result = {}
            result.update(LAYERS)
            result.update(TABLES)
            return result
        else:
            raise ValueError(f"Unknown category: {category}")

    def get_metadata(self, dataset: str) -> dict:
        """Get metadata for a dataset from the API."""
        if dataset in self._metadata_cache:
            return self._metadata_cache[dataset]

        base_url, layer_id = self._get_url_and_id(dataset)
        url = f"{base_url}/{layer_id}"

        self._rate_limit_wait()
        response = self._session.get(url, params={"f": "pjson"}, timeout=self._get_timeout())
        response.raise_for_status()

        metadata = response.json()
        self._metadata_cache[dataset] = metadata
        return metadata

    def get_field_aliases(self, dataset: str) -> dict:
        """
        Get field name to alias mapping from API metadata.

        The API provides human-readable aliases for each field.

        Args:
            dataset: Dataset name

        Returns:
            Dict mapping field names to aliases

        Example:
            >>> aliases = fp.get_field_aliases('wellbore')
            >>> print(aliases['wlbWellboreName'])
            'Wellbore name'
        """
        metadata = self.get_metadata(dataset)
        fields = metadata.get('fields', [])
        return {
            f['name']: f.get('alias', f['name'])
            for f in fields
        }

    def cache_schema(
        self,
        datasets: Optional[list[str]] = None,
        progress: bool = True
    ) -> dict:
        """
        Fetch and cache column metadata from the API.

        This stores column information including:
        - Aliases (human-readable names like 'Wellbore name')
        - Data types
        - Detected foreign key relationships (via alias patterns like 'NPDID wellbore')

        The metadata is stored in columns.json and used for:
        - Displaying readable column names when printing entities
        - Auto-detecting foreign key relationships between tables

        Args:
            datasets: List of datasets to fetch metadata for. If None, uses
                     core entity datasets.
            progress: Show progress messages

        Returns:
            Dict mapping dataset names to column count

        Example:
            >>> # Cache schema for core datasets
            >>> fp.cache_schema()
            >>>
            >>> # Now entity printing uses readable names
            >>> print(troll.wells)  # Shows "Wellbore name" instead of "wlbWellboreName"
            >>>
            >>> # And foreign keys are auto-detected
            >>> fks = fp.schema.get_foreign_keys_by_alias('field_licensee')
            >>> print(fks)  # {'fldNpdidField': 'field', 'cmpNpdidCompany': 'company'}
        """
        from .entities import clear_alias_cache

        if datasets is None:
            datasets = ['field', 'discovery', 'wellbore', 'facility', 'company',
                       'licence', 'field_licensee', 'licence_licensee_history']

        results = {}
        for dataset in datasets:
            try:
                if progress:
                    print(f"Fetching schema for {dataset}...")
                metadata = self.get_metadata(dataset)
                fields = metadata.get('fields', [])
                table_config = self._schema.get_table(dataset)
                self._schema.set_columns(dataset, fields, table_config)
                results[dataset] = len(fields)

                # Show detected foreign keys
                if progress:
                    fks = self._schema.get_foreign_keys_by_alias(dataset)
                    if fks:
                        print(f"  Detected {len(fks)} foreign keys: {list(fks.keys())[:3]}...")
            except Exception as e:
                if progress:
                    print(f"  Warning: Could not fetch schema for {dataset}: {e}")

        # Clear the module-level alias cache so it reloads
        clear_alias_cache()

        if progress:
            total = sum(results.values())
            print(f"Cached {total} columns from {len(results)} datasets")

        return results

    def get_count(self, dataset: str, where: str = "1=1") -> int:
        """Get record count for a dataset from the API."""
        base_url, layer_id = self._get_url_and_id(dataset)
        url = f"{base_url}/{layer_id}/query"
        params = {"where": where, "returnCountOnly": "true", "f": "json"}

        # Use cached get to avoid duplicate count requests
        response = self._cached_get(url, params)
        return response.json().get("count", 0)

    def download(
        self,
        dataset: str,
        where: str = "1=1",
        fields: str = "*",
        include_geometry: bool = True,
        max_records: Optional[int] = None,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Download data from the API.

        Args:
            dataset: Dataset name
            where: SQL WHERE clause for filtering
            fields: Comma-separated field names or '*' for all
            include_geometry: Include geometry as GeoJSON
            max_records: Maximum records to download
            progress: Show progress messages

        Returns:
            DataFrame with downloaded data
        """
        base_url, layer_id = self._get_url_and_id(dataset)
        url = f"{base_url}/{layer_id}/query"

        total = self.get_count(dataset, where)
        if progress:
            print(f"Downloading {dataset}: {total:,} records")

        if max_records:
            total = min(total, max_records)

        batch_size = 1000
        all_records: list[dict] = []
        offset = 0

        while offset < total:
            self._rate_limit_wait()

            params = {
                "where": where,
                "outFields": fields,
                "returnGeometry": str(include_geometry).lower(),
                "resultOffset": offset,
                "resultRecordCount": batch_size,
                "f": "geojson"
            }

            response = self._session.get(url, params=params, timeout=self._get_timeout())
            response.raise_for_status()

            data = response.json()
            features = data.get("features", [])

            for feature in features:
                record = feature.get("properties", {})
                if include_geometry and feature.get("geometry"):
                    record["_geometry"] = json.dumps(feature["geometry"])
                all_records.append(record)

            offset += batch_size

            if progress and offset % 5000 == 0:
                print(f"  {min(offset, total):,} / {total:,}")

        if progress:
            print(f"  Done: {len(all_records):,} records")

        return pd.DataFrame(all_records)

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def sync(
        self,
        datasets: Optional[list[str]] = None,
        category: Optional[str] = None,
        force: bool = False,
        progress: bool = True
    ) -> dict:
        """
        Sync datasets from API to local database.

        Args:
            datasets: Specific datasets to sync
            category: Sync all datasets in category ('entities', 'production', etc.)
            force: Force sync even if data is fresh
            progress: Show progress

        Returns:
            Dict with sync results
        """
        from .sync import SyncEngine, IfMissing, AlwaysSync
        from .database import FILE_MAPPING

        engine = SyncEngine(self, self.db)
        strategy = AlwaysSync() if force else IfMissing()

        results = {}

        if category:
            if category not in FILE_MAPPING:
                raise ValueError(f"Unknown category: {category}")
            results[category] = engine.sync_category(category, strategy=strategy, progress=progress)

        elif datasets:
            for dataset in datasets:
                result = engine.sync_dataset(dataset, strategy=strategy, progress=progress)
                results[dataset] = result

        else:
            # Sync core entities by default
            core_datasets = ['discovery', 'field', 'wellbore', 'facility', 'company', 'licence']
            for dataset in core_datasets:
                result = engine.sync_dataset(dataset, strategy=strategy, progress=progress)
                results[dataset] = result

        return results

    def sync_all(self, force: bool = False, progress: bool = True) -> dict:
        """
        Sync all datasets from API to local database.

        Args:
            force: Force sync even if data is fresh
            progress: Show progress

        Returns:
            Dict with sync results
        """
        from .sync import SyncEngine, IfMissing, AlwaysSync

        engine = SyncEngine(self, self.db)
        strategy = AlwaysSync() if force else IfMissing()

        return engine.sync_all(strategy=strategy, progress=progress)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def fields(self, status: Optional[str] = None) -> pd.DataFrame:
        """Get all fields, optionally filtered by status."""
        df = self._ensure_dataset('field')
        if df is None:
            return pd.DataFrame()

        if status:
            df = df[df['fldCurrentActivitySatus'] == status]

        return df

    def discoveries(self, year: Optional[int] = None) -> pd.DataFrame:
        """Get all discoveries, optionally filtered by year."""
        df = self._ensure_dataset('discovery')
        if df is None:
            return pd.DataFrame()

        if year and 'dscDiscoveryYear' in df.columns:
            df = df[df['dscDiscoveryYear'] == year]

        return df

    def wells(self, status: Optional[str] = None) -> pd.DataFrame:
        """Get all wellbores, optionally filtered by status."""
        df = self._ensure_dataset('wellbore')
        if df is None:
            return pd.DataFrame()

        if status:
            df = df[df['wlbStatus'] == status]

        return df

    def status(self) -> None:
        """Print database status."""
        self.db.print_status()
