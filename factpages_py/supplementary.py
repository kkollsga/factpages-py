"""
Supplementary Data Management

Handles side-loading of external data not available in the API.
Key use cases:
- Custom data from FactPages downloads
- Custom data sources (internal databases, spreadsheets)
- Extending the dataset with domain-specific data

IMPORTANT: Sideloaded data is stored separately from API data to prevent
data contamination. All sideloaded datasets use the 'sideload_' prefix
and are stored in a separate subdirectory.

Custom entities are defined via JSON configuration files stored alongside
the sideloaded data. This allows users to customize how entities are
displayed and navigated without modifying the library code.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

from .entity_config import (
    EntityConfig,
    CustomEntity,
    generate_config_template,
    print_config_template,
    load_entity_config,
    save_entity_config,
)

if TYPE_CHECKING:
    from .database import Database


# Dataset name prefix for sideloaded data (must match database.py)
SIDELOAD_PREFIX = "sideload_"
ENTITY_CONFIG_SUFFIX = ".entity.json"


class SupplementaryData:
    """
    Manager for supplementary (side-loaded) data.

    Sideloaded data is stored separately from API data to prevent
    data contamination. All sideloaded datasets are prefixed with
    'sideload_' and stored in a separate subdirectory.

    Custom entities are defined via JSON configuration files. This allows
    users to customize how data is displayed and navigated.

    Example:
        >>> supp = SupplementaryData(db)
        >>>
        >>> # Load JSON data with entity config
        >>> supp.load_json('data.json', entity_config={
        ...     "entity_type": "MyEntity",
        ...     "dataset": "sideload_my_data",
        ...     "id_field": "id",
        ...     "name_field": "name",
        ...     "properties": {"status": "status"},
        ...     "related": {},
        ...     "display": {"title": "{name}"}
        ... })
        >>>
        >>> # Access as custom entity
        >>> item = supp.entity('myentity', 'Item Name')
        >>> print(item)
    """

    def __init__(self, db: "Database"):
        """
        Initialize supplementary data manager.

        Args:
            db: Database instance for storage
        """
        self.db = db
        self._loaded: Dict[str, dict] = {}  # name -> schema info
        self._entity_configs: Dict[str, EntityConfig] = {}  # entity_type -> config

    def _prefixed_name(self, name: str) -> str:
        """Ensure dataset name has sideload prefix."""
        if name.startswith(SIDELOAD_PREFIX):
            return name
        return f"{SIDELOAD_PREFIX}{name}"

    def _extract_records(self, data: Any, key: Optional[str] = None) -> List[dict]:
        """
        Extract records from JSON data.

        Handles various JSON structures:
        - List of records: [{"id": 1}, {"id": 2}]
        - Object with data key: {"data": [...]}
        - Nested structure: {"entities": {"items": [...]}}
        """
        if key and isinstance(data, dict) and key in data:
            return self._extract_records(data[key])

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common keys
            for common_key in ['data', 'records', 'items', 'entities', 'results']:
                if common_key in data and isinstance(data[common_key], list):
                    return data[common_key]

            # Find first list in values
            for value in data.values():
                if isinstance(value, list) and len(value) > 0:
                    return value

            # Single record
            return [data]

        return []

    def load_json(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        key: Optional[str] = None,
        entity_config: Optional[Union[EntityConfig, dict, str, Path]] = None,
        source: str = "sideloaded",
        recursive: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from a JSON file.

        Automatically detects and loads tables from the JSON structure.
        Optionally registers an entity configuration for easy access.

        Args:
            path: Path to JSON file
            name: Name for the main dataset (defaults to filename without extension)
            key: Specific key to extract from JSON (optional)
            entity_config: Entity configuration (dict, path, or EntityConfig).
                          If provided, the entity type will be registered automatically.
            source: Source identifier for tracking
            recursive: If True, recursively load nested tables

        Returns:
            Dict mapping dataset names to DataFrames

        Example:
            >>> # Simple load
            >>> result = supp.load_json('items.json')
            >>> print(result.keys())  # ['items']
            >>>
            >>> # Load with entity config
            >>> result = supp.load_json('items.json', entity_config={
            ...     "entity_type": "Item",
            ...     "dataset": "sideload_items",
            ...     "id_field": "id",
            ...     "name_field": "name",
            ...     "properties": {},
            ...     "related": {},
            ...     "display": {"title": "{name}"}
            ... })
            >>> item = supp.entity('item', 'First Item')
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Default name from filename
        if name is None:
            name = path.stem

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {}

        # Extract main records
        records = self._extract_records(data, key)
        if records:
            df = pd.DataFrame(records)
            full_name = self._prefixed_name(name)
            self.db.put(full_name, df, source=source)
            result[name] = df
            self._loaded[name] = {
                'path': str(path),
                'record_count': len(df),
                'dataset_name': full_name,
            }

        # Recursively load nested tables if enabled
        if recursive and isinstance(data, dict):
            result.update(self._load_nested_tables(data, path, source, name))

        # Register entity config if provided
        if entity_config is not None:
            self.register_entity(entity_config, save=True)

        return result

    def _load_nested_tables(
        self,
        data: dict,
        path: Path,
        source: str,
        parent_name: str,
    ) -> Dict[str, pd.DataFrame]:
        """Recursively load nested tables from JSON structure."""
        result = {}

        for key, value in data.items():
            # Skip if already processed as main data
            if key in ['data', 'records', 'items', 'entities', 'results']:
                continue

            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # This is a table
                table_name = key
                df = pd.DataFrame(value)
                full_name = self._prefixed_name(table_name)
                self.db.put(full_name, df, source=source)
                result[table_name] = df
                self._loaded[table_name] = {
                    'path': str(path),
                    'record_count': len(df),
                    'dataset_name': full_name,
                    'parent': parent_name,
                }

            elif isinstance(value, dict):
                # Recurse into nested objects
                nested = self._load_nested_tables(value, path, source, parent_name)
                result.update(nested)

        return result

    def get(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a supplementary dataset.

        Args:
            name: Dataset name (with or without 'sideload_' prefix)

        Returns:
            DataFrame or None if not found
        """
        full_name = self._prefixed_name(name)
        return self.db.get_or_none(full_name)

    def list(self) -> Dict[str, dict]:
        """List all loaded supplementary datasets."""
        return self._loaded.copy()

    def has(self, name: str) -> bool:
        """Check if a supplementary dataset is loaded."""
        full_name = self._prefixed_name(name)
        return self.db.has_dataset(full_name)

    # =========================================================================
    # Entity Configuration
    # =========================================================================

    def _config_path(self, entity_type: str) -> Path:
        """Get path to entity config file."""
        return self.db.sideload_dir / f"{entity_type.lower()}{ENTITY_CONFIG_SUFFIX}"

    def _load_entity_configs(self) -> None:
        """Load all entity configs from the sideload directory."""
        if not self.db.sideload_dir.exists():
            return

        for config_file in self.db.sideload_dir.glob(f"*{ENTITY_CONFIG_SUFFIX}"):
            try:
                config = load_entity_config(config_file)
                self._entity_configs[config.entity_type.lower()] = config
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load entity config {config_file}: {e}")

    def register_entity(
        self,
        config: Union[EntityConfig, dict, str, Path],
        save: bool = True
    ) -> EntityConfig:
        """
        Register a custom entity configuration.

        Once registered, the entity type appears in list_entities() and
        can be accessed via entity() or fp.custom_entity().

        Args:
            config: EntityConfig object, dict, or path to JSON file
            save: If True, save config to sideload directory (persists across sessions)

        Returns:
            The registered EntityConfig

        Example:
            >>> # From dict
            >>> supp.register_entity({
            ...     "entity_type": "Project",
            ...     "dataset": "sideload_projects",
            ...     "id_field": "project_id",
            ...     "name_field": "project_name",
            ...     "properties": {"status": {"column": "status"}},
            ...     "related": {},
            ...     "display": {"title": "{name}"}
            ... })
            >>>
            >>> # From file
            >>> supp.register_entity('path/to/entity.config.json')
        """
        if isinstance(config, (str, Path)):
            config = load_entity_config(config)
        elif isinstance(config, dict):
            config = EntityConfig.from_dict(config)

        self._entity_configs[config.entity_type.lower()] = config

        if save:
            config_path = self._config_path(config.entity_type)
            save_entity_config(config, config_path)

        return config

    def get_entity_config(self, entity_type: str) -> Optional[EntityConfig]:
        """
        Get entity configuration by type.

        Args:
            entity_type: Entity type name (case-insensitive)

        Returns:
            EntityConfig or None if not found
        """
        # Lazy load configs
        if not self._entity_configs:
            self._load_entity_configs()

        return self._entity_configs.get(entity_type.lower())

    def entity(self, entity_type: str, name: str) -> CustomEntity:
        """
        Get a custom entity by type and name.

        Args:
            entity_type: Entity type (e.g., "project", "asset")
            name: Entity name (case-insensitive search)

        Returns:
            CustomEntity instance

        Example:
            >>> project = supp.entity('project', 'Alpha')
            >>> print(project)
            >>> print(project.tasks)  # If related data configured
        """
        config = self.get_entity_config(entity_type)
        if config is None:
            raise ValueError(
                f"No entity config found for '{entity_type}'. "
                f"Register one with: supp.register_entity(config)"
            )

        # Get the dataset
        df = self.db.get_or_none(config.dataset)
        if df is None or df.empty:
            raise ValueError(
                f"Dataset '{config.dataset}' not found. "
                f"Load data first with load_json()."
            )

        # Search by name (case-insensitive)
        name_col = config.name_field
        name_upper = name.upper()
        match = df[df[name_col].str.upper() == name_upper]

        if match.empty:
            # Try partial match
            match = df[df[name_col].str.upper().str.contains(name_upper, na=False)]

        if match.empty:
            raise ValueError(f"{config.entity_type} '{name}' not found.")

        return CustomEntity(match.iloc[0], self.db, config)

    def list_entities(self) -> List[str]:
        """
        List all registered entity types.

        Returns:
            List of entity type names (lowercase)
        """
        if not self._entity_configs:
            self._load_entity_configs()
        return list(self._entity_configs.keys())

    def create_entity_template(
        self,
        entity_type: str,
        dataset: str,
        save: bool = False
    ) -> EntityConfig:
        """
        Create an entity configuration template from existing data.

        Inspects the dataset to auto-generate property mappings.

        Args:
            entity_type: Name for the entity (e.g., "Project")
            dataset: Dataset name (with or without 'sideload_' prefix)
            save: If True, save template to sideload directory

        Returns:
            EntityConfig template (edit and register to use)

        Example:
            >>> # Generate template from loaded data
            >>> template = supp.create_entity_template("Project", "projects")
            >>> # Customize the template
            >>> template.display.explore_hint = ".tasks  .members"
            >>> # Register it
            >>> supp.register_entity(template)
        """
        full_name = self._prefixed_name(dataset)
        df = self.db.get_or_none(full_name)

        config = generate_config_template(entity_type, full_name, df)

        if save:
            config_path = self._config_path(entity_type)
            save_entity_config(config, config_path)
            print(f"Template saved to: {config_path}")

        return config

    @staticmethod
    def print_entity_template(entity_type: str = "MyEntity", dataset: str = "my_dataset") -> None:
        """
        Print a human-readable entity configuration template.

        Use this as a starting point to create your own entity config.

        Args:
            entity_type: Entity type name
            dataset: Dataset name

        Example:
            >>> SupplementaryData.print_entity_template("Project", "sideload_projects")
        """
        print(print_config_template(entity_type, dataset))
