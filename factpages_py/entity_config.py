"""
Custom Entity Configuration System

Allows users to define custom entities via JSON configuration files.
Entity configs are stored alongside sideloaded data, making them
portable and easy to customize.

Example config:
    {
        "entity_type": "Project",
        "dataset": "sideload_projects",
        "id_field": "project_id",
        "name_field": "project_name",
        "properties": {
            "status": {"column": "status"},
            "region": {"column": "region"},
            "budget": {"column": "budget_mnok", "type": "float", "nullable": true}
        },
        "related": {
            "tasks": {
                "dataset": "sideload_tasks",
                "join_field": "project_id",
                "display_columns": ["task_name", "status", "due_date"]
            }
        },
        "display": {
            "title": "{name}",
            "subtitle": "{status} | {region}",
            "sections": [
                {"label": "Budget", "format": "{budget} MNOK"}
            ],
            "explore_hint": ".tasks"
        }
    }

Usage:
    >>> from factpages_py.entity_config import CustomEntity, load_entity_config
    >>> config = load_entity_config('path/to/project.entity.json')
    >>> project = CustomEntity(row_data, db, config)
    >>> print(project)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from .database import Database


# =============================================================================
# Entity Configuration Schema
# =============================================================================

@dataclass
class PropertyConfig:
    """Configuration for a single property."""
    column: str
    type: str = "string"  # string, int, float, bool, date
    nullable: bool = True
    default: Any = None


@dataclass
class RelatedConfig:
    """Configuration for a related dataset."""
    dataset: str
    join_field: str
    display_columns: list = field(default_factory=list)
    display_name: Optional[str] = None
    # For indirect relationships (e.g., project -> project_member -> employee)
    via_dataset: Optional[str] = None
    via_join_field: Optional[str] = None
    via_target_field: Optional[str] = None
    # For "current" computed value (get latest record)
    current_field: Optional[str] = None  # Field to get latest value from
    sort_by: Optional[str] = None  # Sort field for getting latest
    sort_ascending: bool = False


@dataclass
class DisplaySection:
    """Configuration for a display section."""
    label: str
    property: Optional[str] = None
    format: Optional[str] = None  # Format string with {property} placeholders


@dataclass
class DisplayConfig:
    """Configuration for entity display."""
    title: str = "{name}"
    subtitle: Optional[str] = None
    sections: list = field(default_factory=list)
    explore_hint: Optional[str] = None


@dataclass
class EntityConfig:
    """
    Complete configuration for a custom entity.

    This defines how to read, display, and navigate an entity.
    """
    entity_type: str
    dataset: str
    id_field: str
    name_field: str
    properties: dict  # name -> PropertyConfig
    related: dict  # name -> RelatedConfig
    display: DisplayConfig

    @classmethod
    def from_dict(cls, data: dict) -> "EntityConfig":
        """Create EntityConfig from a dictionary."""
        # Parse properties
        properties = {}
        for name, prop_data in data.get("properties", {}).items():
            if isinstance(prop_data, str):
                # Shorthand: just column name
                properties[name] = PropertyConfig(column=prop_data)
            else:
                properties[name] = PropertyConfig(**prop_data)

        # Parse related datasets
        related = {}
        for name, rel_data in data.get("related", {}).items():
            related[name] = RelatedConfig(**rel_data)

        # Parse display config
        display_data = data.get("display", {})
        sections = []
        for sec in display_data.get("sections", []):
            sections.append(DisplaySection(**sec))
        display = DisplayConfig(
            title=display_data.get("title", "{name}"),
            subtitle=display_data.get("subtitle"),
            sections=sections,
            explore_hint=display_data.get("explore_hint"),
        )

        return cls(
            entity_type=data["entity_type"],
            dataset=data["dataset"],
            id_field=data["id_field"],
            name_field=data["name_field"],
            properties=properties,
            related=related,
            display=display,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "entity_type": self.entity_type,
            "dataset": self.dataset,
            "id_field": self.id_field,
            "name_field": self.name_field,
            "properties": {
                name: {
                    "column": prop.column,
                    "type": prop.type,
                    "nullable": prop.nullable,
                    **({"default": prop.default} if prop.default is not None else {})
                }
                for name, prop in self.properties.items()
            },
            "related": {
                name: {
                    "dataset": rel.dataset,
                    "join_field": rel.join_field,
                    **({"display_columns": rel.display_columns} if rel.display_columns else {}),
                    **({"display_name": rel.display_name} if rel.display_name else {}),
                    **({"via_dataset": rel.via_dataset} if rel.via_dataset else {}),
                    **({"via_join_field": rel.via_join_field} if rel.via_join_field else {}),
                    **({"via_target_field": rel.via_target_field} if rel.via_target_field else {}),
                    **({"current_field": rel.current_field} if rel.current_field else {}),
                    **({"sort_by": rel.sort_by} if rel.sort_by else {}),
                    **({"sort_ascending": rel.sort_ascending} if rel.sort_ascending else {}),
                }
                for name, rel in self.related.items()
            },
            "display": {
                "title": self.display.title,
                **({"subtitle": self.display.subtitle} if self.display.subtitle else {}),
                "sections": [
                    {
                        "label": sec.label,
                        **({"property": sec.property} if sec.property else {}),
                        **({"format": sec.format} if sec.format else {}),
                    }
                    for sec in self.display.sections
                ],
                **({"explore_hint": self.display.explore_hint} if self.display.explore_hint else {}),
            },
        }


def load_entity_config(path: Union[str, Path]) -> EntityConfig:
    """Load entity configuration from a JSON file."""
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return EntityConfig.from_dict(data)


def save_entity_config(config: EntityConfig, path: Union[str, Path]) -> None:
    """Save entity configuration to a JSON file."""
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)


# =============================================================================
# Custom Entity Class
# =============================================================================

class CustomEntity:
    """
    A generic entity that reads its structure from configuration.

    This allows users to define custom entities for sideloaded data
    without modifying the library code.

    Example:
        >>> config = load_entity_config('project.entity.json')
        >>> project = CustomEntity(row_data, db, config)
        >>> print(project.name)
        >>> print(project.tasks)  # Related data
    """

    def __init__(self, data: pd.Series, db: "Database", config: EntityConfig):
        self._data = data
        self._db = db
        self._config = config
        self._related_cache: dict[str, pd.DataFrame] = {}

    @property
    def id(self) -> int:
        """Entity ID."""
        val = self._data.get(self._config.id_field, 0)
        return int(val) if pd.notna(val) else 0

    @property
    def name(self) -> str:
        """Entity name."""
        return str(self._data.get(self._config.name_field, ''))

    @property
    def entity_type(self) -> str:
        """The type of this entity."""
        return self._config.entity_type

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access for properties and related data.

        Properties are read from the row data.
        Related datasets are loaded on demand.
        """
        # Check if it's a property
        if name in self._config.properties:
            return self._get_property(name)

        # Check if it's a related dataset
        if name in self._config.related:
            return self._get_related(name)

        raise AttributeError(f"'{self._config.entity_type}' has no attribute '{name}'")

    def _get_property(self, name: str) -> Any:
        """Get a property value with type conversion."""
        prop_config = self._config.properties[name]
        value = self._data.get(prop_config.column)

        if pd.isna(value):
            return prop_config.default

        # Type conversion
        if prop_config.type == "int":
            return int(value)
        elif prop_config.type == "float":
            return float(value)
        elif prop_config.type == "bool":
            return bool(value)
        elif prop_config.type == "date":
            return str(value)[:10] if value else None
        else:
            return str(value) if value else ""

    def _get_related(self, name: str) -> Any:
        """Get related data as EntityDataFrame."""
        from .entities import EntityDataFrame

        rel_config = self._config.related[name]

        # Check cache
        if name not in self._related_cache:
            self._related_cache[name] = self._load_related(rel_config)

        df = self._related_cache[name]
        display_name = rel_config.display_name or name.replace('_', ' ').title()

        return EntityDataFrame(
            df,
            entity_type=display_name,
            field_name=self.name,
            display_columns=rel_config.display_columns if rel_config.display_columns else None,
        )

    def _load_related(self, rel_config: RelatedConfig) -> pd.DataFrame:
        """Load related data from the database."""
        related_df = self._db.get_or_none(rel_config.dataset)
        if related_df is None:
            return pd.DataFrame()

        # Filter by join field
        filtered = related_df[related_df[rel_config.join_field] == self.id]

        # Handle indirect relationships (via another dataset)
        if rel_config.via_dataset:
            via_df = self._db.get_or_none(rel_config.via_dataset)
            if via_df is not None and not filtered.empty:
                # Get IDs from the intermediate table
                via_ids = filtered[rel_config.via_join_field or rel_config.join_field].tolist()
                # Filter the target dataset
                target_field = rel_config.via_target_field or rel_config.via_join_field
                filtered = via_df[via_df[target_field].isin(via_ids)]

        return filtered

    def get_current(self, related_name: str) -> dict:
        """
        Get the most recent record from a related dataset.

        Useful for things like "current estimate" or "latest status".

        Args:
            related_name: Name of the related dataset

        Returns:
            Dict with the most recent record's values
        """
        if related_name not in self._config.related:
            return {}

        rel_config = self._config.related[related_name]
        df = self._get_related(related_name)

        if df.empty:
            return {}

        # Sort if configured
        if rel_config.sort_by and rel_config.sort_by in df.columns:
            df = df.sort_values(rel_config.sort_by, ascending=rel_config.sort_ascending)

        # Return first row as dict
        return df.iloc[0].to_dict()

    def __repr__(self) -> str:
        return f"{self._config.entity_type}('{self.name}')"

    def __str__(self) -> str:
        lines = []

        # Title
        title = self._format_string(self._config.display.title)
        lines.append(f"\n{self._config.entity_type}: {title}")
        lines.append("=" * 55)

        # Subtitle
        if self._config.display.subtitle:
            subtitle = self._format_string(self._config.display.subtitle)
            lines.append(subtitle)

        # Sections
        for section in self._config.display.sections:
            if section.format:
                value = self._format_string(section.format)
            elif section.property:
                value = str(getattr(self, section.property, ''))
            else:
                continue

            if value:  # Only show non-empty values
                lines.append(f"{section.label}: {value}")

        # Related data summaries
        for name, rel_config in self._config.related.items():
            if rel_config.current_field:
                current = self.get_current(name)
                if current:
                    display_name = rel_config.display_name or name.replace('_', ' ').title()
                    lines.append("")
                    lines.append(f"Current {display_name}:")
                    for col in rel_config.display_columns[:4]:
                        if col in current:
                            val = current[col]
                            if isinstance(val, float):
                                val = f"{val:.2f}"
                            lines.append(f"  {col}: {val}")

        # Explore hint
        if self._config.display.explore_hint:
            lines.append("")
            lines.append(f"Explore: {self._config.display.explore_hint}")

        return '\n'.join(lines)

    def _format_string(self, template: str) -> str:
        """Format a template string with property values."""
        result = template

        # Replace {name} with self.name
        result = result.replace("{name}", self.name)
        result = result.replace("{id}", str(self.id))

        # Replace property placeholders
        for prop_name in self._config.properties:
            placeholder = "{" + prop_name + "}"
            if placeholder in result:
                value = getattr(self, prop_name, '')
                result = result.replace(placeholder, str(value) if value else '')

        return result


# =============================================================================
# Template Generator
# =============================================================================

def generate_config_template(
    entity_type: str,
    dataset: str,
    sample_df: Optional[pd.DataFrame] = None,
) -> EntityConfig:
    """
    Generate a configuration template for a new entity type.

    Args:
        entity_type: Name of the entity (e.g., "Project")
        dataset: Dataset name (e.g., "sideload_projects")
        sample_df: Optional sample DataFrame to infer columns

    Returns:
        EntityConfig with sensible defaults

    Example:
        >>> template = generate_config_template("Project", "sideload_projects", df)
        >>> save_entity_config(template, "project.entity.json")
        >>> # Now edit the JSON file to customize
    """
    # Try to infer id and name fields
    id_field = "id"
    name_field = "name"
    properties = {}

    if sample_df is not None and not sample_df.empty:
        columns = sample_df.columns.tolist()

        # Guess ID field
        for col in columns:
            if 'npdid' in col.lower() or col.lower() == 'id':
                id_field = col
                break

        # Guess name field
        for col in columns:
            if 'name' in col.lower():
                name_field = col
                break

        # Add all columns as properties
        for col in columns:
            if col in (id_field, name_field):
                continue

            # Infer type
            dtype = sample_df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                prop_type = "int"
            elif pd.api.types.is_float_dtype(dtype):
                prop_type = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                prop_type = "bool"
            else:
                prop_type = "string"

            properties[col] = PropertyConfig(
                column=col,
                type=prop_type,
                nullable=sample_df[col].isna().any(),
            )

    return EntityConfig(
        entity_type=entity_type,
        dataset=dataset,
        id_field=id_field,
        name_field=name_field,
        properties=properties,
        related={},
        display=DisplayConfig(
            title="{name}",
            subtitle=None,
            sections=[],
            explore_hint=None,
        ),
    )


def print_config_template(entity_type: str = "MyEntity", dataset: str = "my_dataset") -> str:
    """
    Print a human-readable configuration template.

    Returns:
        JSON string template with comments

    Example:
        >>> print(print_config_template("Project", "sideload_projects"))
    """
    template = {
        "_comment": "Entity configuration file. Customize this for your data.",
        "entity_type": entity_type,
        "dataset": dataset,
        "id_field": "id_column_name",
        "name_field": "name_column_name",
        "properties": {
            "status": {
                "column": "status_column",
                "type": "string",
                "_comment": "type can be: string, int, float, bool, date"
            },
            "value": {
                "column": "numeric_column",
                "type": "float",
                "nullable": True,
                "default": 0.0
            }
        },
        "related": {
            "items": {
                "dataset": "related_dataset_name",
                "join_field": "foreign_key_column",
                "display_columns": ["col1", "col2", "col3"],
                "display_name": "Related Items",
                "_comment": "For indirect joins, add: via_dataset, via_join_field, via_target_field"
            },
            "current_value": {
                "dataset": "history_dataset",
                "join_field": "entity_id",
                "sort_by": "date_column",
                "sort_ascending": False,
                "current_field": "value",
                "_comment": "Use current_field + sort_by to get latest record"
            }
        },
        "display": {
            "title": "{name}",
            "subtitle": "{status} | {value}",
            "sections": [
                {"label": "Status", "property": "status"},
                {"label": "Location", "format": "Block {block_number}"}
            ],
            "explore_hint": ".items  .current_value"
        }
    }
    return json.dumps(template, indent=2)
