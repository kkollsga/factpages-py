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

    def field(self, identifier: Union[str, int]) -> Field:
        """
        Get a Field entity by name or ID.

        Automatically loads all field_* tables (reserves, licensees, etc.)
        when auto_sync is enabled.

        Args:
            identifier: Field name (case-insensitive) or npdid (int)

        Returns:
            Field object with properties and methods

        Example:
            >>> troll = fp.field("troll")  # By name
            >>> troll = fp.field(43506)    # By npdid
            >>> print(troll.operator)
            >>> print(troll.reserves)
        """
        fields = self._ensure_dataset('field')
        if fields is None:
            raise ValueError("Field data not available. Run sync() first.")

        # Auto-load related tables (field_reserves, field_licensee_hst, etc.)
        self._ensure_related_datasets('field')

        if isinstance(identifier, int):
            # Search by npdid
            match = fields[fields['fldNpdidField'] == identifier]
            if match.empty:
                raise ValueError(f"Field with npdid {identifier} not found.")
        else:
            # Case-insensitive name search
            name_upper = identifier.upper()
            match = fields[fields['fldName'].str.upper() == name_upper]

            if match.empty:
                # Try partial match
                match = fields[fields['fldName'].str.upper().str.contains(name_upper, na=False)]

            if match.empty:
                raise ValueError(f"Field '{identifier}' not found.")

        return Field(match.iloc[0], self.db)

    def discovery(self, identifier: Union[str, int]) -> Discovery:
        """
        Get a Discovery entity by name or ID.

        Automatically loads all discovery_* tables when auto_sync is enabled.

        Args:
            identifier: Discovery name (case-insensitive) or npdid (int)

        Returns:
            Discovery object

        Example:
            >>> johan = fp.discovery("johan sverdrup")  # By name
            >>> johan = fp.discovery(28851043)         # By npdid
        """
        discoveries = self._ensure_dataset('discovery')
        if discoveries is None:
            raise ValueError("Discovery data not available. Run sync() first.")

        # Auto-load related tables (discovery_reserves, discovery_operator_hst, etc.)
        self._ensure_related_datasets('discovery')

        if isinstance(identifier, int):
            # Search by npdid
            match = discoveries[discoveries['dscNpdidDiscovery'] == identifier]
            if match.empty:
                raise ValueError(f"Discovery with npdid {identifier} not found.")
        else:
            # Case-insensitive name search
            name_upper = identifier.upper()
            match = discoveries[discoveries['dscName'].str.upper() == name_upper]

            if match.empty:
                match = discoveries[discoveries['dscName'].str.upper().str.contains(name_upper, na=False)]

            if match.empty:
                raise ValueError(f"Discovery '{identifier}' not found.")

        return Discovery(match.iloc[0], self.db)

    def well(self, identifier: Union[str, int]) -> Wellbore:
        """
        Get a Wellbore entity by name or ID.

        Automatically loads all wellbore_* tables when auto_sync is enabled.

        Args:
            identifier: Wellbore name (e.g., '35/11-25') or npdid (int)

        Returns:
            Wellbore object

        Example:
            >>> well = fp.well("31/2-1")    # By name
            >>> well = fp.well(1234567)     # By npdid
        """
        wellbores = self._ensure_dataset('wellbore')
        if wellbores is None:
            raise ValueError("Wellbore data not available. Run sync() first.")

        # Auto-load related tables (wellbore_core, wellbore_dst, etc.)
        self._ensure_related_datasets('wellbore')

        if isinstance(identifier, int):
            # Search by npdid
            match = wellbores[wellbores['wlbNpdidWellbore'] == identifier]
            if match.empty:
                raise ValueError(f"Wellbore with npdid {identifier} not found.")
        else:
            # Name search
            match = wellbores[wellbores['wlbWellboreName'] == identifier]

            if match.empty:
                # Try partial match
                match = wellbores[wellbores['wlbWellboreName'].str.contains(identifier, case=False, na=False)]

            if match.empty:
                raise ValueError(f"Wellbore '{identifier}' not found.")

        return Wellbore(match.iloc[0], self.db)

    # Alias for well()
    def wellbore(self, identifier: Union[str, int]) -> Wellbore:
        """Alias for well(). See well() for documentation."""
        return self.well(identifier)

    def company(self, identifier: Union[str, int]) -> Company:
        """
        Get a Company entity by name or ID.

        Args:
            identifier: Company name (partial match) or npdid (int)

        Returns:
            Company object

        Example:
            >>> equinor = fp.company("equinor")  # By name
            >>> equinor = fp.company(561234)     # By npdid
        """
        companies = self._ensure_dataset('company')
        if companies is None:
            raise ValueError("Company data not available. Run sync() first.")

        if isinstance(identifier, int):
            # Search by npdid
            match = companies[companies['cmpNpdidCompany'] == identifier]
            if match.empty:
                raise ValueError(f"Company with npdid {identifier} not found.")
        else:
            # Name search (partial match)
            match = companies[companies['cmpLongName'].str.contains(identifier, case=False, na=False)]

            if match.empty:
                raise ValueError(f"Company matching '{identifier}' not found.")

        return Company(match.iloc[0], self.db)

    def license(self, identifier: Union[str, int]) -> License:
        """
        Get a License entity by name or ID.

        Automatically loads all licence_* tables when auto_sync is enabled.

        Args:
            identifier: License name (e.g., 'PL001') or npdid (int)

        Returns:
            License object

        Example:
            >>> pl001 = fp.license("PL001")  # By name
            >>> pl001 = fp.license(1234567)  # By npdid
        """
        licences = self._ensure_dataset('licence')
        if licences is None:
            raise ValueError("License data not available. Run sync() first.")

        # Auto-load related tables (licence_licensee_hst, licence_operator_hst, etc.)
        self._ensure_related_datasets('licence')

        if isinstance(identifier, int):
            # Search by npdid
            match = licences[licences['prlNpdidLicence'] == identifier]
            if match.empty:
                raise ValueError(f"License with npdid {identifier} not found.")
        else:
            # Name search
            match = licences[licences['prlName'] == identifier]

            if match.empty:
                match = licences[licences['prlName'].str.contains(identifier, case=False, na=False)]

            if match.empty:
                raise ValueError(f"License '{identifier}' not found.")

        return License(match.iloc[0], self.db)

    def facility(self, identifier: Union[str, int]) -> Facility:
        """
        Get a Facility entity by name or ID.

        Args:
            identifier: Facility name or npdid (int)

        Returns:
            Facility object

        Example:
            >>> troll_a = fp.facility("TROLL A")  # By name
            >>> troll_a = fp.facility(1234567)    # By npdid
        """
        facilities = self._ensure_dataset('facility')
        if facilities is None:
            raise ValueError("Facility data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = facilities[facilities['fclNpdidFacility'] == identifier]
            if match.empty:
                raise ValueError(f"Facility with npdid {identifier} not found.")
        else:
            match = facilities[facilities['fclName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Facility matching '{identifier}' not found.")

        return Facility(match.iloc[0], self.db)

    def pipeline(self, identifier: Union[str, int]) -> Pipeline:
        """
        Get a Pipeline entity by name or ID.

        Args:
            identifier: Pipeline name or npdid (int)

        Returns:
            Pipeline object

        Example:
            >>> pipe = fp.pipeline("STATPIPE")  # By name
            >>> pipe = fp.pipeline(1234567)     # By npdid
        """
        pipelines = self._ensure_dataset('pipeline')
        if pipelines is None:
            raise ValueError("Pipeline data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = pipelines[pipelines['pipNpdidPipeline'] == identifier]
            if match.empty:
                raise ValueError(f"Pipeline with npdid {identifier} not found.")
        else:
            match = pipelines[pipelines['pipName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Pipeline matching '{identifier}' not found.")

        return Pipeline(match.iloc[0], self.db)

    def play(self, identifier: Union[str, int]) -> Play:
        """
        Get a Play entity by name or ID.

        Args:
            identifier: Play name or npdid (int)

        Returns:
            Play object

        Example:
            >>> play = fp.play("UPPER JURASSIC")  # By name
            >>> play = fp.play(1234567)           # By npdid
        """
        plays = self._ensure_dataset('play')
        if plays is None:
            raise ValueError("Play data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = plays[plays['plyNpdidPlay'] == identifier]
            if match.empty:
                raise ValueError(f"Play with npdid {identifier} not found.")
        else:
            match = plays[plays['plyName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Play matching '{identifier}' not found.")

        return Play(match.iloc[0], self.db)

    def block(self, identifier: Union[str, int]) -> Block:
        """
        Get a Block entity by name or ID.

        Args:
            identifier: Block name (e.g., '34/10') or npdid (int)

        Returns:
            Block object

        Example:
            >>> block = fp.block("34/10")     # By name
            >>> block = fp.block(1234567)     # By npdid
        """
        blocks = self._ensure_dataset('block')
        if blocks is None:
            raise ValueError("Block data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = blocks[blocks['blkNpdidBlock'] == identifier]
            if match.empty:
                raise ValueError(f"Block with npdid {identifier} not found.")
        else:
            match = blocks[blocks['blkName'] == identifier]
            if match.empty:
                match = blocks[blocks['blkName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Block '{identifier}' not found.")

        return Block(match.iloc[0], self.db)

    def quadrant(self, identifier: Union[str, int]) -> Quadrant:
        """
        Get a Quadrant entity by name or ID.

        Args:
            identifier: Quadrant name (e.g., '34') or npdid (int)

        Returns:
            Quadrant object

        Example:
            >>> quad = fp.quadrant("34")      # By name
            >>> quad = fp.quadrant(1234567)   # By npdid
        """
        quadrants = self._ensure_dataset('quadrant')
        if quadrants is None:
            raise ValueError("Quadrant data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = quadrants[quadrants['quaNpdidQuadrant'] == identifier]
            if match.empty:
                raise ValueError(f"Quadrant with npdid {identifier} not found.")
        else:
            match = quadrants[quadrants['quaName'] == identifier]
            if match.empty:
                match = quadrants[quadrants['quaName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Quadrant '{identifier}' not found.")

        return Quadrant(match.iloc[0], self.db)

    def tuf(self, identifier: Union[str, int]) -> TUF:
        """
        Get a TUF (onshore facility) entity by name or ID.

        Args:
            identifier: TUF name or npdid (int)

        Returns:
            TUF object

        Example:
            >>> kollsnes = fp.tuf("KOLLSNES")  # By name
            >>> kollsnes = fp.tuf(1234567)     # By npdid
        """
        tufs = self._ensure_dataset('tuf')
        if tufs is None:
            raise ValueError("TUF data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = tufs[tufs['tufNpdidTuf'] == identifier]
            if match.empty:
                raise ValueError(f"TUF with npdid {identifier} not found.")
        else:
            match = tufs[tufs['tufName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"TUF matching '{identifier}' not found.")

        return TUF(match.iloc[0], self.db)

    def seismic(self, identifier: Union[str, int]) -> Seismic:
        """
        Get a Seismic survey entity by name or ID.

        Args:
            identifier: Survey name or npdid (int)

        Returns:
            Seismic object

        Example:
            >>> survey = fp.seismic("NPD-1901")  # By name
            >>> survey = fp.seismic(1234567)     # By npdid
        """
        surveys = self._ensure_dataset('seismic_acquisition')
        if surveys is None:
            raise ValueError("Seismic data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = surveys[surveys['seisNpdidSurvey'] == identifier]
            if match.empty:
                raise ValueError(f"Seismic survey with npdid {identifier} not found.")
        else:
            match = surveys[surveys['seisSurveyName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Seismic survey matching '{identifier}' not found.")

        return Seismic(match.iloc[0], self.db)

    def stratigraphy(self, identifier: Union[str, int]) -> Stratigraphy:
        """
        Get a Stratigraphy entity by name or ID.

        Args:
            identifier: Formation name or npdid (int)

        Returns:
            Stratigraphy object

        Example:
            >>> draupne = fp.stratigraphy("DRAUPNE")  # By name
            >>> draupne = fp.stratigraphy(1234567)    # By npdid
        """
        # Try lithostratigraphy first
        litho = self._ensure_dataset('strat_litho')
        if litho is not None and not litho.empty:
            if isinstance(identifier, int):
                match = litho[litho['lsuNpdidLithoStrat'] == identifier]
            else:
                match = litho[litho['lsuName'].str.contains(identifier, case=False, na=False)]
            if not match.empty:
                return Stratigraphy(match.iloc[0], self.db)

        # Fall back to chronostratigraphy
        chrono = self._ensure_dataset('strat_chrono')
        if chrono is not None and not chrono.empty:
            if isinstance(identifier, int):
                match = chrono[chrono['strNpdidChronoStrat'] == identifier]
            else:
                match = chrono[chrono['strName'].str.contains(identifier, case=False, na=False)]
            if not match.empty:
                return Stratigraphy(match.iloc[0], self.db)

        raise ValueError(f"Stratigraphy '{identifier}' not found.")

    def business_arrangement(self, identifier: Union[str, int]) -> BusinessArrangement:
        """
        Get a Business Arrangement entity by name or ID.

        Args:
            identifier: Arrangement name or npdid (int)

        Returns:
            BusinessArrangement object

        Example:
            >>> ba = fp.business_arrangement("TROLL UNIT")  # By name
            >>> ba = fp.business_arrangement(1234567)       # By npdid
        """
        arrangements = self._ensure_dataset('business_arrangement_area')
        if arrangements is None:
            raise ValueError("Business arrangement data not available. Run sync() first.")

        if isinstance(identifier, int):
            match = arrangements[arrangements['baaNpdidBsnsArrArea'] == identifier]
            if match.empty:
                raise ValueError(f"Business arrangement with npdid {identifier} not found.")
        else:
            match = arrangements[arrangements['baaName'].str.contains(identifier, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Business arrangement matching '{identifier}' not found.")

        return BusinessArrangement(match.iloc[0], self.db)

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

        if store and not df.empty:
            self.db.put(dataset, df, source="api")

        return df

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def sync(
        self,
        datasets: Optional[Union[str, list[str]]] = None,
        category: Optional[str] = None,
        force: bool = False,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Sync datasets from API to local database.

        Args:
            datasets: Specific dataset(s) to sync (string or list)
            category: Sync all datasets in category ('entities', 'production', etc.)
            force: Force sync even if data is fresh
            progress: Show progress
            workers: Number of parallel download threads (default: 4)

        Returns:
            Dict with sync results

        Example:
            >>> fp.sync('field')  # Single table
            >>> fp.sync(['field', 'discovery'])  # Multiple tables
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .sync import SyncEngine, IfMissing, AlwaysSync
        from .database import FILE_MAPPING

        # Handle string input (single dataset)
        if isinstance(datasets, str):
            datasets = [datasets]

        engine = SyncEngine(self, self.db)
        strategy = AlwaysSync() if force else IfMissing()

        results = {}

        if category:
            if category not in FILE_MAPPING:
                raise ValueError(f"Unknown category: {category}")
            results[category] = engine.sync_category(category, strategy=strategy, progress=progress)

        else:
            # Determine which datasets to sync
            if datasets:
                to_sync = datasets
            else:
                # Sync core entities by default
                to_sync = ['discovery', 'field', 'wellbore', 'facility', 'company', 'licence']

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
            if not to_download:
                return results

            if progress:
                print(f"Downloading {len(to_download)} tables...")

            # Track completed downloads
            completed = 0

            def sync_one(dataset: str) -> tuple[str, dict]:
                result = engine.sync_dataset(dataset, strategy=strategy, progress=False)
                return dataset, result

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(sync_one, ds): ds for ds in to_download}

                for future in as_completed(futures):
                    dataset, result = future.result()
                    results[dataset] = result
                    completed += 1
                    if progress:
                        count = result.get('record_count', 0)
                        print(f"  [{completed}/{len(to_download)}] {dataset}: {count:,} records")

        return results

    def sync_all(self, force: bool = False, progress: bool = True, workers: int = 4) -> dict:
        """
        Sync all datasets from API to local database.

        Args:
            force: Force sync even if data is fresh
            progress: Show progress
            workers: Number of parallel download threads (default: 4)

        Returns:
            Dict with sync results
        """
        from .sync import SyncEngine, IfMissing, AlwaysSync

        engine = SyncEngine(self, self.db)
        strategy = AlwaysSync() if force else IfMissing()

        return engine.sync_all(strategy=strategy, progress=progress, workers=workers)

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

    def refresh(
        self,
        max_age_days: int = 30,
        limit_percent: float = 10.0,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Refresh stale datasets with a limit on how many to download.

        Designed for regular maintenance - refreshes the oldest datasets
        first, but limits downloads to avoid overwhelming the API.
        Run this periodically (e.g., weekly) to keep data fresh.

        Args:
            max_age_days: Consider datasets older than this stale (default: 30)
            limit_percent: Maximum percentage of datasets to refresh (default: 10%)
            progress: Show progress messages
            workers: Number of parallel download threads

        Returns:
            Dict with refresh results

        Example:
            >>> # Refresh up to 10% of datasets older than 30 days
            >>> results = fp.refresh()
            >>>
            >>> # More aggressive: refresh up to 25%
            >>> results = fp.refresh(limit_percent=25)
            >>>
            >>> # Check remaining stale datasets
            >>> print(f"Still stale: {results['stale_remaining']}")
        """
        from .sync import SyncEngine
        engine = SyncEngine(self, self.db)
        return engine.refresh(
            max_age_days=max_age_days,
            limit_percent=limit_percent,
            progress=progress,
            workers=workers
        )

    def fix(
        self,
        max_age_days: int = 30,
        include_missing: bool = True,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Thorough fix: refresh ALL stale and missing datasets without limits.

        Use this when you need a complete data refresh, like after a long
        period of inactivity or when data quality is critical. Unlike
        refresh(), this has no limit on how many datasets to download.

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
                        f"Run fp.sync('{name}') to download it."
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
