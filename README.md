# sodir-api

Python client for the Norwegian Offshore Directorate (Sodir) REST API.

Access petroleum data from the Norwegian Continental Shelf including wellbores, discoveries, fields, production data, licensing, and more.

## Installation

```bash
pip install sodir-api
```

## Quick Start

```python
from sodir_api import SodirAPI

api = SodirAPI()

# Download all discoveries
discoveries = api.download('discovery')
print(f"Found {len(discoveries)} discoveries")

# Filter by hydrocarbon type
oil_discoveries = api.download('discovery', where="dscHcType='OIL'")

# Get monthly production data
production = api.download('field_production_monthly')

# Search with keyword arguments
producing_fields = api.search('field', fldCurrentActivitySatus='Producing')
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
# List all available datasets
api.list_datasets()

# Get metadata for a dataset
api.summary('discovery')

# Get field definitions
fields = api.get_fields('wellbore')
```

## Features

- **Simple API**: Download data in one line of code
- **Pandas Integration**: Returns DataFrames ready for analysis
- **Geometry Support**: Optionally include GeoJSON geometries
- **Filtering**: SQL WHERE clause support for server-side filtering
- **Pagination**: Automatically handles large datasets
- **Rate Limiting**: Built-in rate limiting to be nice to the server

## Examples

### Download with Filtering

```python
# Discoveries since 2010
recent = api.download('discovery', where="dscDiscoveryYear >= 2010")

# Specific fields only
df = api.download(
    'wellbore',
    fields='wlbWellboreName,wlbTotalDepth,wlbWaterDepth',
    include_geometry=False
)
```

### Multiple Datasets

```python
# Download several datasets at once
data = api.download_many(['discovery', 'field', 'wellbore'])

discoveries = data['discovery']
fields = data['field']
wellbores = data['wellbore']
```

### Generate Inventory

```python
# Generate complete inventory with all field definitions
inventory = api.generate_inventory('sodir_inventory.json')
```

## API Reference

### SodirAPI

```python
api = SodirAPI(timeout=30, rate_limit=0.2)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `list_datasets(category)` | List available datasets |
| `get_metadata(dataset)` | Get dataset metadata |
| `get_fields(dataset)` | Get field definitions as DataFrame |
| `get_count(dataset, where)` | Get record count |
| `download(dataset, ...)` | Download data as DataFrame |
| `download_many(datasets)` | Download multiple datasets |
| `search(dataset, **filters)` | Search with keyword filters |
| `summary(dataset)` | Print dataset summary |
| `generate_inventory(file)` | Generate complete inventory |

## License

MIT License

## Acknowledgments

Data provided by the [Norwegian Offshore Directorate](https://www.sodir.no/) (Sodir).
