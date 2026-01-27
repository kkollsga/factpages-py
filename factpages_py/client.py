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
from __future__ import annotations

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

from datetime import datetime, timezone
from .datasets import LAYERS, TABLES, FACTMAPS_LAYERS, METADATA_BASE_URL, METADATA_TABLES
from .database import Database
from .parallel import parallel_sync, parallel_count
from .analysis import AnalysisMixin


def _convert_unix_ts(val):
    """Convert Unix timestamp (ms) to simplest appropriate format.

    Returns: date, datetime, or datetime with ms - whichever is simplest.
    """
    if pd.isna(val) or val is None:
        return None
    try:
        ms = int(val) % 1000
        ts_seconds = val / 1000
        dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)

        # Midnight with no ms → date only
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and ms == 0:
            return dt.strftime('%Y-%m-%d')
        # Has time but no ms → datetime
        if ms == 0:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        # Has ms → include them
        return dt.strftime('%Y-%m-%d %H:%M:%S') + f'.{ms:03d}'
    except (ValueError, OSError, OverflowError, TypeError):
        return None


def _is_likely_timestamp_column(col_name: str) -> bool:
    """Check if column name suggests it contains timestamps."""
    col_lower = col_name.lower()
    # Common date/time patterns in column names
    patterns = ['date', 'updated', 'created', 'time', 'timestamp']
    return any(p in col_lower for p in patterns)


def _convert_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Unix timestamp columns (milliseconds) to ISO date strings.

    Detects columns by name patterns (date, updated, created, time) and by
    value range (13-digit integers typical of Unix ms timestamps).

    Args:
        df: DataFrame with potential timestamp columns

    Returns:
        DataFrame with converted date columns
    """
    if df.empty:
        return df

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue

        first_val = sample.iloc[0]

        # Check if value looks like Unix timestamp in ms (13 digits, > 1e12)
        try:
            val = float(first_val)
            # Valid range: 2001-01-01 to 2100-01-01 in ms
            if 9.78e11 < val < 4.1e12:
                # Either column name suggests date, or it's a likely timestamp
                if _is_likely_timestamp_column(col) or (val > 1e12 and val < 2e12):
                    df[col] = df[col].apply(_convert_unix_ts)
        except (ValueError, TypeError):
            pass

    return df
from .entities import (
    Field, Discovery, Wellbore, Company, License, Entity,
    Facility, Pipeline, Play, Block, Quadrant, TUF, Seismic,
    Stratigraphy, BusinessArrangement
)
from .entity_config import CustomEntity
from .graph import GraphEndpoints
from .supplementary import SupplementaryData
from .schema import SchemaRegistry


# =============================================================================
# Entity Accessor
# =============================================================================

class EntityAccessor:
    """
    Provides convenient access to entities with list(), ids(), and random selection.

    Example:
        >>> fp.field("troll")      # Get by name
        >>> fp.field(43506)        # Get by ID
        >>> fp.field()             # Get random field
        >>> fp.field.list()        # List all field names
        >>> fp.field.ids()         # List all field IDs
        >>> fp.field.count()       # Count of fields
        >>> fp.field.all()         # DataFrame of all fields
    """

    def __init__(
        self,
        client: "Factpages",
        dataset: str,
        entity_class: type,
        id_column: str,
        name_column: str,
        alt_datasets: Optional[list[str]] = None,
    ):
        self._client = client
        self._dataset = dataset
        self._entity_class = entity_class
        self._id_column = id_column
        self._name_column = name_column
        self._alt_datasets = alt_datasets or []

    def _get_df(self) -> Optional[pd.DataFrame]:
        """Get the entity DataFrame."""
        df = self._client._ensure_dataset(self._dataset)
        if df is None or df.empty:
            # Try alternative datasets
            for alt in self._alt_datasets:
                df = self._client._ensure_dataset(alt)
                if df is not None and not df.empty:
                    return df
        return df

    def __call__(self, identifier: Optional[Union[str, int]] = None):
        """
        Get entity by name, ID, or random if no argument provided.

        Args:
            identifier: Entity name (case-insensitive) or npdid (int).
                       If None, returns a random entity.

        Returns:
            Entity instance

        Example:
            >>> fp.field("troll")   # By name
            >>> fp.field(43506)     # By ID
            >>> fp.field()          # Random field
        """
        import random

        df = self._get_df()
        if df is None or df.empty:
            raise ValueError(
                f"{self._entity_class.__name__} data not available. Run sync() first."
            )

        # If no identifier, return random entity
        if identifier is None:
            idx = random.randint(0, len(df) - 1)
            return self._entity_class(df.iloc[idx], self._client.db)

        # Search by ID
        if isinstance(identifier, int):
            match = df[df[self._id_column] == identifier]
            if match.empty:
                raise ValueError(
                    f"{self._entity_class.__name__} with npdid {identifier} not found."
                )
            return self._entity_class(match.iloc[0], self._client.db)

        # Search by name (case-insensitive)
        name_upper = identifier.upper()
        match = df[df[self._name_column].str.upper() == name_upper]

        if match.empty:
            # Try partial match
            match = df[df[self._name_column].str.upper().str.contains(name_upper, na=False)]

        if match.empty:
            raise ValueError(f"{self._entity_class.__name__} '{identifier}' not found.")

        return self._entity_class(match.iloc[0], self._client.db)

    def list(self) -> list[str]:
        """
        List all entity names.

        Returns:
            Sorted list of entity names

        Example:
            >>> fp.field.list()
            ['ALBUSKJELL', 'ALVHEIM', 'BALDER', ...]
        """
        df = self._get_df()
        if df is None or df.empty:
            return []
        return sorted(df[self._name_column].dropna().unique().tolist())

    def ids(self) -> list[int]:
        """
        List all entity IDs.

        Returns:
            List of entity npdids

        Example:
            >>> fp.field.ids()
            [43437, 43506, 43548, ...]
        """
        df = self._get_df()
        if df is None or df.empty:
            return []
        return df[self._id_column].dropna().astype(int).tolist()

    def count(self) -> int:
        """
        Count of entities.

        Returns:
            Number of entities

        Example:
            >>> fp.field.count()
            141
        """
        df = self._get_df()
        if df is None or df.empty:
            return 0
        return len(df)

    def all(self) -> pd.DataFrame:
        """
        Get all entities as DataFrame.

        Returns:
            DataFrame of all entities

        Example:
            >>> fp.field.all()
        """
        df = self._get_df()
        if df is None:
            return pd.DataFrame()
        return df

    def __repr__(self) -> str:
        count = self.count()
        return f"<{self._entity_class.__name__}Accessor: {count} entities>"


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
    timeout: int = 30  # Read timeout (between bytes)
    connect_timeout: int = 10  # Connection timeout
    total_timeout: int = 120  # Total request timeout (2 min default per table)

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


class Factpages(AnalysisMixin):
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

        # Entity accessors - provide list(), ids(), count(), random access
        # Names match table names for consistency
        self.field = EntityAccessor(
            self, 'field', Field, 'fldNpdidField', 'fldName'
        )
        self.discovery = EntityAccessor(
            self, 'discovery', Discovery, 'dscNpdidDiscovery', 'dscName'
        )
        self.wellbore = EntityAccessor(
            self, 'wellbore', Wellbore, 'wlbNpdidWellbore', 'wlbWellboreName'
        )
        self.company = EntityAccessor(
            self, 'company', Company, 'cmpNpdidCompany', 'cmpLongName'
        )
        self.licence = EntityAccessor(
            self, 'licence', License, 'prlNpdidLicence', 'prlName'
        )
        self.facility = EntityAccessor(
            self, 'facility', Facility, 'fclNpdidFacility', 'fclName'
        )
        self.pipeline = EntityAccessor(
            self, 'pipeline', Pipeline, 'pipNpdidPipeline', 'pipName'
        )
        self.play = EntityAccessor(
            self, 'play', Play, 'plyNpdidPlay', 'plyName'
        )
        self.block = EntityAccessor(
            self, 'block', Block, 'blkNpdidBlock', 'blkName'
        )
        self.quadrant = EntityAccessor(
            self, 'quadrant', Quadrant, 'quaNpdidQuadrant', 'quaName'
        )
        self.tuf = EntityAccessor(
            self, 'tuf', TUF, 'tufNpdidTuf', 'tufName'
        )
        self.seismic = EntityAccessor(
            self, 'seismic_acquisition', Seismic, 'seisNpdidSurvey', 'seisSurveyName'
        )
        self.stratigraphy = EntityAccessor(
            self, 'strat_litho', Stratigraphy, 'lsuNpdidLithoStrat', 'lsuName',
            alt_datasets=['strat_chrono']
        )
        self.business_arrangement = EntityAccessor(
            self, 'business_arrangement_area', BusinessArrangement,
            'baaNpdidBsnsArrArea', 'baaName'
        )

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
    # Entity accessors are initialized in __init__ as EntityAccessor instances.
    # Accessor names match table names: field, discovery, wellbore, company,
    # licence, facility, pipeline, play, block, quadrant, tuf, seismic,
    # stratigraphy, business_arrangement
    #
    # They support:
    #   - fp.field("troll")     # Get by name
    #   - fp.field(43506)       # Get by ID
    #   - fp.field()            # Random entity
    #   - fp.field.list()       # List all names
    #   - fp.field.ids()        # List all IDs
    #   - fp.field.count()      # Count
    #   - fp.field.all()        # DataFrame

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
            df = self.download(dataset)
            self.db.put(dataset, df, source='api')
            return df

        return None

    def _ensure_related_datasets(self, prefix: str) -> None:
        """
        Ensure all datasets with a given prefix are loaded.

        When accessing an entity like 'field', this loads all field_* tables
        (field_reserves, field_licensee_hst, etc.) in parallel.

        Args:
            prefix: Dataset prefix (e.g., 'field', 'wellbore', 'discovery', 'licence')
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Find all datasets with this prefix
        all_datasets = {**LAYERS, **TABLES}
        related = [name for name in all_datasets if name.startswith(f"{prefix}_")]

        # Filter to only datasets we don't already have
        to_load = [ds for ds in related if not self.db.has_dataset(ds)]

        if not to_load or not self.auto_sync:
            return

        # Load in parallel
        def load_one(dataset: str) -> tuple[str, bool]:
            try:
                df = self.download(dataset)
                self.db.put(dataset, df, source='api')
                return dataset, True
            except Exception:
                return dataset, False

        print(f"Auto-syncing {len(to_load)} {prefix}_* tables...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(load_one, ds): ds for ds in to_load}
            for future in as_completed(futures):
                pass  # Just wait for completion

    # =========================================================================
    # Raw Data API
    # =========================================================================

    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_timeout(self):
        """Get urllib3 Timeout object with connect, read, and total timeouts."""
        from urllib3.util.timeout import Timeout
        return Timeout(
            connect=self.config.connect_timeout,
            read=self.config.timeout,
            total=self.config.total_timeout
        )

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

    def list_tables(self, filter: Optional[str] = None) -> list[str]:
        """
        List downloaded/synced table names.

        Args:
            filter: Optional string to filter table names (e.g., 'field' returns
                   all tables with 'field' in the name)

        Returns:
            Sorted list of downloaded table names

        Example:
            >>> fp.list_tables()
            ['field', 'discovery', 'wellbore']

            >>> fp.list_tables('field')
            ['field', 'field_reserves', 'field_licensee_hst']
        """
        all_tables = self.db.list_datasets()

        if filter:
            all_tables = [t for t in all_tables if filter.lower() in t.lower()]

        return sorted(all_tables)

    def api_tables(self, filter: Optional[str] = None) -> list[str]:
        """
        List tables available on the API.

        Args:
            filter: Optional string to filter table names (e.g., 'field' returns
                   all tables with 'field' in the name)

        Returns:
            Sorted list of API table names

        Example:
            >>> fp.api_tables()
            ['block', 'company', 'discovery', ...]

            >>> fp.api_tables('field')
            ['field', 'field_activity_status_hst', 'field_description', ...]
        """
        all_tables = list(LAYERS.keys()) + list(TABLES.keys())

        if filter:
            all_tables = [t for t in all_tables if filter.lower() in t.lower()]

        return sorted(all_tables)

    def df(self, table: str) -> pd.DataFrame:
        """
        Get a table as a pandas DataFrame.

        If auto_sync is enabled and the table isn't downloaded, it will be
        fetched from the API automatically.

        Args:
            table: Table name (e.g., 'field', 'field_reserves')

        Returns:
            pandas DataFrame

        Example:
            >>> fp.df('field')
            >>> fp.df('field_reserves')
        """
        return self._ensure_dataset(table)

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
        store: bool = False,
    ) -> pd.DataFrame:
        """
        Download data from the API.

        Args:
            dataset: Dataset name
            where: SQL WHERE clause for filtering
            fields: Comma-separated field names or '*' for all
            include_geometry: Include geometry as GeoJSON
            max_records: Maximum records to download
            store: If True, store downloaded data in local database

        Returns:
            DataFrame with downloaded data
        """
        base_url, layer_id = self._get_url_and_id(dataset)
        url = f"{base_url}/{layer_id}/query"

        total = self.get_count(dataset, where)

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

        df = pd.DataFrame(all_records)

        # Convert Unix timestamp columns to ISO date strings
        df = _convert_timestamp_columns(df)

        if store and not df.empty:
            self.db.put(dataset, df, source="api")

        return df

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def refresh(
        self,
        datasets: Optional[Union[str, list[str]]] = None,
        category: Optional[str] = None,
        force: bool = False,
        max_stale_days: int = 25,
        limit_percent: float = 10.0,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Refresh datasets from API to local database.

        When called without arguments, performs maintenance:
        - Uses cached stats to identify datasets needing refresh
        - Fetches datasets with row count mismatch (high priority)
        - Fetches stale datasets (older than max_stale_days)
        - Limited to limit_percent of total datasets by default

        When called with specific datasets, downloads those datasets.

        Args:
            datasets: Specific dataset(s) to refresh. Options:
                - None: Maintenance mode (fix mismatches + stale data)
                - "all": All available datasets (75+ tables)
                - "stats": Force refresh of stats/metadata only
                - "field": Single table
                - ["field", "discovery"]: Multiple tables
            category: Refresh all datasets in category ('entities', 'production', etc.)
            force: Force refresh even if data exists locally
            max_stale_days: Consider datasets older than this stale (default: 25)
            limit_percent: Max percent of datasets to refresh in maintenance mode (default: 10)
            progress: Show progress
            workers: Number of parallel download threads (default: 4)

        Returns:
            Dict with refresh results

        Example:
            >>> fp.refresh()  # Maintenance: uses cached stats, fix mismatches + stale
            >>> fp.refresh('all')  # Download missing tables only
            >>> fp.refresh('all', force=True)  # Force re-download all tables
            >>> fp.refresh('stats')  # Force refresh stats/metadata
            >>> fp.refresh('field')  # Single table
            >>> fp.refresh('field', force=True)  # Force re-download single table
        """
        from .sync import SyncEngine, IfMissing, AlwaysSync
        from .database import FILE_MAPPING

        engine = SyncEngine(self, self.db)

        # Handle "stats" special keyword - force metadata/stats refresh
        if datasets == "stats":
            if progress:
                print("Refreshing stats and metadata...")
            result = self._fetch_metadata(progress=progress, force=True)
            # Also refresh stats cache
            self.stats(progress=progress, force_refresh=True, workers=workers)
            return {'status': 'stats_refreshed', **result}

        # Handle "all" special keyword - smart check then download missing
        if datasets == "all":
            all_datasets = list({**LAYERS, **TABLES}.keys())

            if force:
                # Force mode: download everything
                self._fetch_metadata(progress=progress)
                return engine.sync_all(strategy=AlwaysSync(), progress=progress, workers=workers)

            # Check what's already cached
            missing = [ds for ds in all_datasets if not self.db.has_dataset(ds)]

            if not missing:
                return {
                    'status': 'ok',
                    'cached': len(all_datasets),
                    'downloaded': 0,
                    'tables': {}
                }

            # Fetch metadata for downloads (to get expected counts)
            self._fetch_metadata(progress=False)
                        # Download only missing datasets with tqdm progress
            results = {
                'status': 'ok',
                'downloaded': [],
                'cached': len(all_datasets) - len(missing),
                'tables': {}
            }

            # Use parallel_sync for consistent parallel fetching
            sync_results = parallel_sync(
                datasets=missing,
                sync_fn=lambda ds: engine.sync_dataset(ds, strategy=AlwaysSync(), progress=False),
                workers=workers,
                progress=progress,
                desc="Downloading",
                get_expected_fn=lambda ds: self.db.get_remote_count(ds) or 0
            )

            # Populate results structure
            for dataset, metadata in sync_results.items():
                results['tables'][dataset] = metadata
                if metadata['status'] == 'synced':
                    results['downloaded'].append(dataset)

            return results

        # Ensure API metadata is available for informative error messages
        self._fetch_metadata(progress=progress)

        # Strategy for remaining cases
        strategy = AlwaysSync() if force else IfMissing()

        # Handle string input (single dataset)
        if isinstance(datasets, str):
            datasets = [datasets]

        results = {}

        if category:
            if category not in FILE_MAPPING:
                raise ValueError(f"Unknown category: {category}")
            results[category] = engine.sync_category(category, strategy=strategy, progress=progress)

        elif datasets:
            # Specific datasets requested - download them
            to_sync = datasets

            # Filter to only datasets that need downloading (unless force)
            if force:
                to_download = to_sync
            else:
                to_download = [ds for ds in to_sync if not self.db.has_dataset(ds)]

            # Mark already-cached as skipped
            for ds in to_sync:
                if ds not in to_download:
                    results[ds] = {'status': 'skipped', 'reason': 'already cached'}

            # Only download if there's something to download
            if to_download:
                # Use parallel_sync for consistent parallel fetching
                sync_results = parallel_sync(
                    datasets=to_download,
                    sync_fn=lambda ds: engine.sync_dataset(ds, strategy=strategy, progress=False),
                    workers=workers,
                    progress=progress,
                    desc="Downloading",
                    get_expected_fn=lambda ds: self.db.get_remote_count(ds) or 0
                )
                results.update(sync_results)

        else:
            # Maintenance mode: ensure core entities + fix mismatches and stale datasets
            import time as time_module
            start_time = time_module.time()

            core_entities = ['field', 'discovery', 'wellbore', 'facility', 'company', 'licence']
            missing_core = [ds for ds in core_entities if not self.db.has_dataset(ds)]

            # Get stats - this returns pre-computed changed/stale lists
            stats = self.stats(progress=False, workers=workers)
            all_datasets = list({**LAYERS, **TABLES}.keys())
            total_count = len(all_datasets)
            max_to_refresh = max(1, int(total_count * limit_percent / 100))

            # Use pre-computed lists from stats (much faster than recomputing)
            # Priority 1: Row count mismatches (changed datasets)
            changed_list = stats.get('changed', [])
            mismatched = [(item['dataset'], abs(item.get('remote_count', 0) - item.get('local_count', 0)))
                          for item in changed_list if item.get('remote_count', 0) > 0]
            mismatched.sort(key=lambda x: x[1], reverse=True)
            mismatched_names = [ds for ds, _ in mismatched]

            # Priority 2: Stale datasets from stats (already computed)
            stale_list = stats.get('stale', [])
            # Filter by max_stale_days (stats uses 7 days, we might want different threshold)
            stale = [(item['dataset'], item.get('age_days', 0))
                     for item in stale_list
                     if item['dataset'] not in mismatched_names
                     and (item.get('age_days') or 0) >= max_stale_days]
            stale.sort(key=lambda x: x[1], reverse=True)
            stale_names = [ds for ds, _ in stale]

            # Combine: core first, then mismatches, then stale up to limit
            to_fetch = missing_core.copy()
            to_fetch.extend(mismatched_names)
            remaining_slots = max_to_refresh - len(mismatched_names)
            if remaining_slots > 0:
                to_fetch.extend(stale_names[:remaining_slots])

            if not to_fetch:
                return {
                    'status': 'ok',
                    'message': 'No datasets need refreshing',
                    'tables': {}
                }

            # Use parallel_sync for consistent parallel fetching
            sync_results = parallel_sync(
                datasets=to_fetch,
                sync_fn=lambda ds: engine.sync_dataset(ds, strategy=AlwaysSync(), progress=False),
                workers=workers,
                progress=progress,
                desc="Refreshing",
                get_expected_fn=lambda ds: self.db.get_remote_count(ds) or 0
            )

            # Add priority field to results
            for dataset, metadata in sync_results.items():
                priority = 'core' if dataset in missing_core else ('mismatch' if dataset in mismatched_names else 'stale')
                metadata['priority'] = priority
                results[dataset] = metadata

            total_elapsed = time_module.time() - start_time

            results['_summary'] = {
                'core_fetched': len(missing_core),
                'mismatched_count': len(mismatched_names),
                'stale_count': len(stale_names),
                'refreshed_count': len(to_fetch),
                'limit_applied': len(to_fetch) >= max_to_refresh,
                'elapsed_seconds': round(total_elapsed, 2)
            }

        return results

    def stats(
        self,
        progress: bool = True,
        workers: int = 4,
        force_refresh: bool = False
    ) -> dict:
        """
        Get statistics for all datasets from the API without downloading data.

        Stats are cached for 3 days to minimize API calls. Use force_refresh=True
        to fetch fresh stats regardless of cache age.

        Fetches only record counts, comparing local vs remote to identify
        datasets that may have changed or are missing. Useful for seeing
        what's available and planning syncs.

        Args:
            progress: Show progress messages
            workers: Number of parallel API requests
            force_refresh: Force refetch from API even if cache is fresh

        Returns:
            Dict with stats including:
            - total_remote_records: Total records available on API
            - changed: Datasets where local count differs from remote
            - missing: Datasets not downloaded yet
            - all: Full list with per-dataset details

        Example:
            >>> stats = fp.stats()
            >>> print(f"Total remote records: {stats['total_remote_records']:,}")
            >>> print(f"Datasets with changes: {len(stats['changed'])}")
        """
        from .sync import SyncEngine
        engine = SyncEngine(self, self.db)
        return engine.stats(progress=progress, workers=workers, force_refresh=force_refresh)

    def check_quality(self, progress: bool = True) -> dict:
        """
        Check data quality and freshness across all datasets.

        Returns a comprehensive report on dataset freshness, including
        a health score and lists of fresh, aging, stale, and missing datasets.

        Args:
            progress: Show progress messages

        Returns:
            Dict with quality report including:
            - health_score: 0-100 score (higher = fresher data)
            - fresh: Datasets < 7 days old
            - aging: Datasets 7-30 days old
            - stale: Datasets > 30 days old
            - missing: Datasets not downloaded

        Example:
            >>> report = fp.check_quality()
            >>> print(f"Health: {report['health_score']}%")
            >>> print(f"Stale datasets: {report['stale_count']}")
        """
        from .sync import SyncEngine
        engine = SyncEngine(self, self.db)
        return engine.check_quality(progress=progress)

    def fix(
        self,
        max_age_days: int = 30,
        include_missing: bool = True,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Fix data quality: refresh ALL stale and missing datasets.

        Identifies datasets older than max_age_days and re-downloads them.
        Use this for data maintenance when you need fresh data across
        the entire database.

        Args:
            max_age_days: Consider datasets older than this stale (default: 30)
            include_missing: Also download missing datasets (default: True)
            progress: Show progress messages
            workers: Number of parallel download threads

        Returns:
            Dict with fix results

        Example:
            >>> # Fix all stale and missing data
            >>> results = fp.fix()
            >>> print(f"Fixed {results['synced_count']} datasets")
            >>>
            >>> # Fix only stale (don't download new datasets)
            >>> results = fp.fix(include_missing=False)
        """
        from .sync import SyncEngine
        engine = SyncEngine(self, self.db)
        return engine.fix(
            max_age_days=max_age_days,
            include_missing=include_missing,
            progress=progress,
            workers=workers
        )

    def fetch_all(
        self,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Fetch the entire database by downloading all missing datasets.

        This method:
        1. Force refreshes stats from the API (ignores cache)
        2. Downloads all datasets that don't exist locally

        Use this for initial setup or to ensure you have all available data.

        Args:
            progress: Show progress messages
            workers: Number of parallel download threads

        Returns:
            Dict with fetch results including:
            - synced: List of successfully downloaded datasets
            - synced_count: Number of datasets downloaded
            - failed: List of datasets that failed to download
            - already_had: Number of datasets already in database
            - total_datasets: Total number of available datasets

        Example:
            >>> # Download entire database
            >>> results = fp.fetch_all()
            >>> print(f"Downloaded {results['synced_count']} datasets")
            >>>
            >>> # Check what was downloaded
            >>> print(f"Already had: {results['already_had']}")
            >>> print(f"Newly downloaded: {results['synced_count']}")
        """
        from .sync import SyncEngine
        engine = SyncEngine(self, self.db)
        return engine.fetch_all(progress=progress, workers=workers)

    # =========================================================================
    # API Metadata
    # =========================================================================

    def _fetch_metadata(self, progress: bool = True, force: bool = False) -> dict:
        """
        Fetch table/column descriptions and remote record counts.

        This is called automatically before data downloads to enable
        informative error messages and download validation.

        Args:
            progress: Show progress messages
            force: Force refetch even if metadata exists

        Returns:
            Dict with fetch results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Check freshness - metadata refreshes every 30 days, counts every 3 days
        metadata_fresh = False
        counts_fresh = False

        if not force:
            if self.db.has_metadata():
                age = self.db.get_metadata_age_days()
                metadata_fresh = age is not None and age < 30

            if self.db.has_remote_counts():
                age = self.db.get_remote_counts_age_days()
                counts_fresh = age is not None and age < 3

        if metadata_fresh and counts_fresh:
            return {'status': 'cached', 'metadata_age_days': self.db.get_metadata_age_days(),
                    'counts_age_days': self.db.get_remote_counts_age_days()}

        results = {'tables': 0, 'columns': 0, 'counts': 0, 'errors': []}

        # Determine what to fetch
        fetch_metadata = not metadata_fresh or force
        fetch_counts = not counts_fresh or force

        if progress:
            parts = []
            if fetch_metadata:
                parts.append("metadata")
            if fetch_counts:
                parts.append("counts")
            print(f"Fetching API {' & '.join(parts)}...", end="", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}

                if fetch_metadata:
                    # Fetch table descriptions (Table 10)
                    futures[executor.submit(self._fetch_table_descriptions)] = 'tables'
                    # Fetch column descriptions (Table 11)
                    futures[executor.submit(self._fetch_column_descriptions)] = 'columns'

                if fetch_counts:
                    # Fetch remote counts for all tables
                    futures[executor.submit(self._fetch_all_remote_counts)] = 'counts'

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result = future.result()
                        results[key] = result
                    except Exception as e:
                        results['errors'].append(f"{key}: {e}")

            if progress:
                parts = []
                if results['tables']:
                    parts.append(f"{results['tables']} tables")
                if results['columns']:
                    parts.append(f"{results['columns']} columns")
                if results['counts']:
                    parts.append(f"{results['counts']} counts")
                print(f" {', '.join(parts)}")

            results['status'] = 'fetched'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))
            if progress:
                print(f" error: {e}")

        return results

    def _fetch_table_descriptions(self) -> int:
        """Fetch table descriptions from Metadata FeatureServer."""
        table_url = f"{METADATA_BASE_URL}/{METADATA_TABLES['table_descriptions']}/query"
        params = {'where': '1=1', 'outFields': '*', 'f': 'json'}

        response = self._session.get(table_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'features' in data:
            descriptions = [f['attributes'] for f in data['features']]
            return self.db.save_table_descriptions(descriptions)
        return 0

    def _fetch_column_descriptions(self) -> int:
        """Fetch column descriptions from Metadata FeatureServer."""
        col_url = f"{METADATA_BASE_URL}/{METADATA_TABLES['attribute_descriptions']}/query"
        params = {'where': '1=1', 'outFields': '*', 'f': 'json'}

        response = self._session.get(col_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'features' in data:
            descriptions = [f['attributes'] for f in data['features']]
            return self.db.save_column_descriptions(descriptions)
        return 0

    def _fetch_all_remote_counts(self) -> int:
        """Fetch remote record counts for all tables in parallel."""
        all_datasets = list({**LAYERS, **TABLES}.keys())

        # Use centralized parallel_count for consistent handling
        counts = parallel_count(
            datasets=all_datasets,
            count_fn=self.get_count,
            workers=8
        )

        return self.db.save_remote_counts(counts)

    def describe(self, table_name: str, column: Optional[str] = None) -> str:
        """
        Get description for a table or column from API metadata.

        Args:
            table_name: Table name (e.g., 'field', 'wellbore')
            column: Optional column name (e.g., 'fldName', 'wlbTotalDepth')

        Returns:
            Description string

        Example:
            >>> fp.describe('field')
            'Fields on the Norwegian continental shelf where oil and/or gas is being or will be produced'

            >>> fp.describe('wellbore', 'wlbTotalDepth')
            'Total depth: Total vertical depth of the wellbore in meters'
        """
        # Ensure we have metadata
        if not self.db.has_metadata():
            self._fetch_metadata(progress=False)

        if column:
            info = self.db.get_column_description(table_name, column)
            if info:
                display = info.get('display_text', column)
                desc = info.get('description', 'No description available')
                dtype = info.get('data_type', '')
                return f"{display}: {desc}" + (f" [{dtype}]" if dtype else "")
            return f"No description found for column '{column}' in table '{table_name}'"

        desc = self.db.get_table_description(table_name)
        return desc if desc else f"No description found for table '{table_name}'"

    def column_info(self, table_name: str) -> pd.DataFrame:
        """
        Get all column descriptions for a table as a DataFrame.

        Args:
            table_name: Table name

        Returns:
            DataFrame with columns: column_name, display_text, data_type, description

        Example:
            >>> fp.column_info('field')
                    column_name    display_text  data_type  description
            0       fldName        Field name    nvarchar   Name of the field
            1       fldNpdidField  NPDID Field   int        Unique identifier...
        """
        # Ensure we have metadata
        if not self.db.has_metadata():
            self._fetch_metadata(progress=False)

        cols = self.db.get_all_column_descriptions(table_name)
        if not cols:
            return pd.DataFrame(columns=['column_name', 'display_text', 'data_type', 'description'])

        rows = [
            {'column_name': name, **info}
            for name, info in cols.items()
        ]
        return pd.DataFrame(rows)

    def tables(self, downloaded_only: bool = False) -> pd.DataFrame:
        """
        Get a list of all available tables with metadata and download status.

        Returns a comprehensive overview of all tables in the API including:
        - Table name and ID
        - Description from metadata
        - Whether the table is downloaded locally
        - Local record count and last sync date

        Args:
            downloaded_only: If True, only show tables that are downloaded locally

        Returns:
            DataFrame with columns:
            - table_name: Name of the table
            - table_id: API layer/table ID
            - description: Human-readable description
            - downloaded: Whether table exists locally
            - local_count: Number of records in local database
            - expected_count: Expected count from API (for comparison)
            - last_sync: Date of last sync (or None)
            - type: 'layer' (has geometry) or 'table' (no geometry)

        Example:
            >>> fp.tables()
                table_name  table_id  description                          downloaded  local_count  expected_count  last_sync   type
            0   field       7100      Fields on the Norwegian...           True        141          141             2024-01-15  layer
            1   wellbore    5000      Primary table for wellbores...       True        9731         9731            2024-01-15  layer
            2   company     1200      Company information                  False       0            185             None        table

            >>> # Show only downloaded tables
            >>> fp.tables(downloaded_only=True)

            >>> # Find tables with count mismatches
            >>> t = fp.tables()
            >>> t[t['local_count'] != t['expected_count']]
        """
        # Ensure we have metadata and counts
        if not self.db.has_metadata() or not self.db.has_remote_counts():
            self._fetch_metadata(progress=False)

        # Get all available tables from LAYERS and TABLES
        all_tables = []

        for name, table_id in LAYERS.items():
            all_tables.append({
                'table_name': name,
                'table_id': table_id,
                'type': 'layer'
            })

        for name, table_id in TABLES.items():
            all_tables.append({
                'table_name': name,
                'table_id': table_id,
                'type': 'table'
            })

        # Enrich with metadata and local status
        rows = []
        for table in all_tables:
            name = table['table_name']

            # Get description from metadata
            description = self.db.get_table_description(name)

            # Get expected count from remote counts
            expected_count = self.db.get_remote_count(name)

            # Check local status
            downloaded = self.db.has_dataset(name)
            local_count = 0
            last_sync = None

            if downloaded:
                local_count = self.db.get_record_count(name)
                sync_time = self.db.get_last_sync(name)
                if sync_time:
                    last_sync = sync_time.strftime('%Y-%m-%d')

            if downloaded_only and not downloaded:
                continue

            rows.append({
                'table_name': name,
                'table_id': table['table_id'],
                'description': description or '',
                'downloaded': downloaded,
                'local_count': local_count,
                'expected_count': expected_count,
                'last_sync': last_sync,
                'type': table['type']
            })

        df = pd.DataFrame(rows)

        # Sort by table_name
        if not df.empty:
            df = df.sort_values('table_name').reset_index(drop=True)

        return df

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

    def wellbores(self, status: Optional[str] = None) -> pd.DataFrame:
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

    # =========================================================================
    # Dynamic Table Access
    # =========================================================================

    def __getattr__(self, name: str):
        """
        Enable dynamic table access via attribute-style calls.

        Allows accessing any table by name and creating Entity objects.
        Works for tables that are in the local database.

        Args:
            name: Table name (e.g., 'field_reserves', 'wellbore_dst')

        Returns:
            A callable that takes an ID and returns an Entity

        Example:
            >>> reserves = fp.field_reserves(43506)  # Get by npdid
            >>> print(reserves.fldRecoverableOil)
            >>>
            >>> dst = fp.wellbore_dst(1234567)  # Get wellbore DST by id
        """
        # Avoid recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Check if this table exists in the database
        if self.db.has_dataset(name):
            def get_entity(id_value: int) -> Entity:
                df = self.db.get(name)
                row = Entity.find_by_id(df, id_value)
                if row is None:
                    # List available ID columns for the error message
                    id_cols = Entity.find_id_columns(df)
                    raise ValueError(
                        f"No row with ID {id_value} found in '{name}'. "
                        f"Searched columns: {id_cols}"
                    )
                return Entity(row, self.db, name)
            return get_entity

        # Check if table is available in the API but not downloaded
        all_datasets = {**LAYERS, **TABLES}
        if name in all_datasets:
            def get_entity_auto(id_value: int) -> Entity:
                df = self._ensure_dataset(name)
                if df is None:
                    raise ValueError(
                        f"Table '{name}' not available. "
                        f"Run fp.refresh('{name}') to download it."
                    )
                row = Entity.find_by_id(df, id_value)
                if row is None:
                    id_cols = Entity.find_id_columns(df)
                    raise ValueError(
                        f"No row with ID {id_value} found in '{name}'. "
                        f"Searched columns: {id_cols}"
                    )
                return Entity(row, self.db, name)
            return get_entity_auto

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def generate_api_inventory(
        self,
        output_path: Optional[Union[str, Path]] = None,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Generate a complete API inventory JSON file with all tables and columns.

        Fetches metadata from the API for all datasets and saves to a JSON file.
        This is useful for documentation and understanding the full data model.

        Args:
            output_path: Path for output JSON (default: api_inventory.json in data dir)
            progress: Show progress
            workers: Number of parallel workers for fetching metadata

        Returns:
            Dict with the complete inventory

        Example:
            >>> fp = Factpages()
            >>> inventory = fp.generate_api_inventory()
            >>> print(f"Found {len(inventory['tables'])} tables")
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json

        if output_path is None:
            output_path = self.db.data_dir / "api_inventory.json"
        else:
            output_path = Path(output_path)

        # Collect all datasets
        all_datasets = {}
        all_datasets.update({name: ("layer", layer_id) for name, layer_id in LAYERS.items()})
        all_datasets.update({name: ("table", table_id) for name, table_id in TABLES.items()})

        inventory = {
            "_comment": "Complete API inventory from Sodir FactMaps REST API",
            "_generated": pd.Timestamp.now().isoformat(),
            "_api_url": self.DATASERVICE_URL,
            "tables": {}
        }

        if progress:
            print(f"Fetching metadata for {len(all_datasets)} tables...")

        completed = 0

        def fetch_one(name: str, ds_type: str, ds_id: int) -> tuple[str, dict]:
            try:
                metadata = self.get_metadata(name)
                fields = metadata.get('fields', [])

                table_info = {
                    "id": ds_id,
                    "type": ds_type,
                    "name": metadata.get('name', name),
                    "description": metadata.get('description', ''),
                    "has_geometry": metadata.get('geometryType') is not None,
                    "geometry_type": metadata.get('geometryType'),
                    "record_count": None,
                    "columns": {}
                }

                # Try to get record count
                try:
                    table_info["record_count"] = self.get_count(name)
                except Exception:
                    pass

                # Process columns
                for field in fields:
                    col_name = field.get('name', '')
                    table_info["columns"][col_name] = {
                        "alias": field.get('alias', col_name),
                        "type": field.get('type', ''),
                        "nullable": field.get('nullable', True),
                        "length": field.get('length'),
                    }

                return name, table_info
            except Exception as e:
                return name, {"error": str(e), "id": ds_id, "type": ds_type}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_one, name, ds_type, ds_id): name
                for name, (ds_type, ds_id) in all_datasets.items()
            }

            for future in as_completed(futures):
                name, table_info = future.result()
                inventory["tables"][name] = table_info
                completed += 1
                if progress and completed % 10 == 0:
                    print(f"  [{completed}/{len(all_datasets)}] {name}")

        # Sort tables alphabetically
        inventory["tables"] = dict(sorted(inventory["tables"].items()))

        # Add summary statistics
        successful = [t for t in inventory["tables"].values() if "error" not in t]
        total_columns = sum(len(t.get("columns", {})) for t in successful)
        total_records = sum(t.get("record_count", 0) or 0 for t in successful)

        inventory["_summary"] = {
            "total_tables": len(all_datasets),
            "successful": len(successful),
            "failed": len(all_datasets) - len(successful),
            "total_columns": total_columns,
            "total_records": total_records,
            "layers_with_geometry": len([t for t in successful if t.get("has_geometry")]),
            "tables_without_geometry": len([t for t in successful if not t.get("has_geometry")])
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(inventory, f, indent=2)

        if progress:
            print(f"\nInventory saved to: {output_path}")
            print(f"Tables: {len(successful)}/{len(all_datasets)}")
            print(f"Columns: {total_columns}")
            print(f"Records: {total_records:,}")

        return inventory
