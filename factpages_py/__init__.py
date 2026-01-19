"""
factpages_py: Python library for Norwegian Petroleum Factpages data

Access petroleum data from the Norwegian Continental Shelf including:
- Fields, discoveries, wellbores, and facilities
- Production data and reserves
- Licensing and ownership information
- Seismic survey coverage
- Stratigraphy and geological data

Quick Start:
    >>> from factpages_py import Factpages
    >>>
    >>> # Initialize client
    >>> fp = Factpages()
    >>>
    >>> # Refresh data from API
    >>> fp.refresh()
    >>>
    >>> # User-friendly entity access
    >>> troll = fp.field("troll")
    >>> print(troll.operator)
    >>> print(troll.partners)
    >>> print(troll.production(2025, 8))
    >>>
    >>> # Raw data access
    >>> fields_df = fp.db.get('field')
    >>>
    >>> # Graph building (for rusty-graph)
    >>> nodes = fp.graph.nodes('field')
    >>> connections = fp.graph.connections('discovery', 'field')

Custom Entities (for sideloaded data):
    >>> # Load external JSON data with entity config
    >>> fp.load_json('projects.json', entity_config={
    ...     "entity_type": "Project",
    ...     "dataset": "sideload_projects",
    ...     "id_field": "id",
    ...     "name_field": "name",
    ...     "properties": {},
    ...     "related": {},
    ...     "display": {"title": "{name}"}
    ... })
    >>>
    >>> # Access as custom entity
    >>> project = fp.custom_entity('project', 'Alpha')
    >>> print(project)
"""

from .client import Factpages, ClientConfig
from .database import Database
from .datasets import LAYERS, TABLES, FACTMAPS_LAYERS
from .entity_config import EntityConfig, CustomEntity, generate_config_template
from .entities import (
    Field, Discovery, Wellbore, Company, License, Entity,
    Facility, Pipeline, Play, Block, Quadrant, TUF, Seismic,
    Stratigraphy, BusinessArrangement
)

__version__ = "0.1.0"
__author__ = "Norwegian Petroleum Data Community"
__all__ = [
    # Client
    "Factpages",
    "ClientConfig",
    "Database",
    # Datasets
    "LAYERS",
    "TABLES",
    "FACTMAPS_LAYERS",
    # Entity configuration
    "EntityConfig",
    "CustomEntity",
    "generate_config_template",
    # Entity classes
    "Field",
    "Discovery",
    "Wellbore",
    "Company",
    "License",
    "Entity",
    "Facility",
    "Pipeline",
    "Play",
    "Block",
    "Quadrant",
    "TUF",
    "Seismic",
    "Stratigraphy",
    "BusinessArrangement",
]
