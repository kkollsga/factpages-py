# factpages-py

Python library for Norwegian Petroleum Factpages data.

Access petroleum data from the Norwegian Continental Shelf including wellbores, discoveries, fields, production data, licensing, and more.

## Installation

```bash
pip install factpages-py
```

## Quick Start

```python
from factpages_py import Factpages

fp = Factpages()

# Sync data from API
fp.sync()

# Access entities
troll = fp.field("troll")
print(troll.operator)
print(troll.partners)

# Get production data
production = troll.production(2025, 8)

# Raw DataFrame access
fields_df = fp.db.get('field')
discoveries_df = fp.db.get('discovery')
```

## Available Datasets

The API provides access to 100+ datasets:

| Category | Examples | Records |
|----------|----------|---------|
| **Core Entities** | discovery, field, wellbore, facility | 12K+ |
| **Production** | field_production_monthly, field_reserves | 10K+ |
| **Wellbore Details** | wellbore_chrono_strat, wellbore_core_photo | 200K+ |
| **Licensing** | licence, licence_licensee_history | 130K+ |
| **Seismic** | seismic_acquisition, seismic_acquisition_3d | 60K+ |
| **Stratigraphy** | strat_litho, strat_chrono | 300+ |

```python
# List all available tables
fp.db.list_tables()

# Get specific dataset
wellbores = fp.db.get('wellbore')
```

## Features

- **Simple API**: Download and access data easily
- **Pandas Integration**: Returns DataFrames ready for analysis
- **Entity Access**: High-level entity objects with properties and relationships
- **Graph Export**: Export data for knowledge graph construction
- **Geometry Support**: Includes GeoJSON geometries where available
- **Local Caching**: Data cached locally in Parquet format

## Graph Integration

Export data for knowledge graph libraries like rusty-graph:

```python
from factpages_py import Factpages
import rusty_graph

fp = Factpages()
fp.sync()

graph = rusty_graph.KnowledgeGraph()

# One-liner bulk loading
export = fp.graph.export_for_graph()
graph.add_nodes_bulk(export['nodes'])
graph.add_connections_from_source(export['connections'])
```

## Examples

### Entity Access

```python
# Get field by name
troll = fp.field("troll")
print(f"Operator: {troll.operator}")
print(f"Status: {troll.status}")

# Get wellbore
wellbore = fp.wellbore("31/2-1")
print(f"Depth: {wellbore.total_depth}m")
```

### Raw Data Access

```python
# Get DataFrame directly
fields = fp.db.get('field')
discoveries = fp.db.get('discovery')

# Filter data (note: API has typo 'Satus' instead of 'Status')
producing = fields[fields['fldCurrentActivitySatus'] == 'Producing']
```

### Sync Specific Tables

```python
# Sync only specific tables
fp.sync(['field', 'discovery', 'wellbore'])

# Sync all tables
fp.sync()
```

## License

MIT License

## Acknowledgments

Data provided by the [Norwegian Offshore Directorate](https://www.sodir.no/) (Sodir).
