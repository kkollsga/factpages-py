# factpages-py

Python library for accessing Norwegian Petroleum Directorate (Sodir) FactPages data.

Access comprehensive petroleum data from the Norwegian Continental Shelf including fields, discoveries, wellbores, facilities, licenses, production data, and more.

## Installation

```bash
pip install factpages-py
```

## Quick Start

```python
from factpages_py import Factpages

# Initialize client with data directory
fp = Factpages(data_dir="./factpages_data")

# Download core datasets
fp.sync()

# Access a field
troll = fp.field("troll")
print(troll)
```

---

## Setup & Data Synchronization

### Setting Up the Database

```python
from factpages_py import Factpages

# Initialize with custom data directory (default: ./factpages_data)
fp = Factpages(data_dir="./my_petroleum_data")

# The database is stored as SQLite at: ./my_petroleum_data/factpages.db
```

### Sync Methods

#### Basic Sync (Core Datasets)

```python
# Sync core entities: field, discovery, wellbore, facility, company, licence
fp.sync()

# Sync specific datasets
fp.sync('field')
fp.sync(['field', 'discovery', 'wellbore'])

# Force re-download even if data exists
fp.sync('field', force=True)
```

#### Full Database Download

```python
# Download ALL available datasets (75+ tables)
results = fp.fetch_all()
print(f"Downloaded {results['synced_count']} datasets")
```

#### Check API Statistics

```python
# Get stats for all datasets (cached for 3 days)
stats = fp.stats()
print(f"Total remote records: {stats['total_remote_records']:,}")
print(f"Missing datasets: {len(stats['missing'])}")
print(f"Changed datasets: {len(stats['changed'])}")

# Force refresh stats from API
stats = fp.stats(force_refresh=True)
```

#### Refresh Stale Data

```python
# Refresh up to 10% of datasets older than 30 days
results = fp.refresh()

# More aggressive refresh
results = fp.refresh(max_age_days=14, limit_percent=25)
```

#### Fix All Data Issues

```python
# Download all stale and missing datasets (no limit)
results = fp.fix()
print(f"Fixed {results['synced_count']} datasets")

# Fix only stale (don't download new datasets)
results = fp.fix(include_missing=False)
```

#### Check Data Quality

```python
report = fp.check_quality()
print(f"Health Score: {report['health_score']}%")
print(f"Fresh (<7d): {report['fresh_count']}")
print(f"Stale (>30d): {report['stale_count']}")
print(f"Missing: {report['missing_count']}")
```

---

## Entity Access

The library provides 14 entity types with rich object-oriented access:

### Fields

```python
troll = fp.field("troll")           # By name (case-insensitive)
troll = fp.field(43506)             # By npdid

print(troll)                        # Formatted display
print(troll.name)                   # TROLL
print(troll.operator)               # Equinor Energy AS
print(troll.status)                 # Producing
print(troll.hc_type)                # OIL/GAS
print(troll.discovery_year)         # 1979
```

### Discoveries

```python
sverdrup = fp.discovery("johan sverdrup")
sverdrup = fp.discovery(28851043)

print(sverdrup)
print(sverdrup.name)
print(sverdrup.status)
print(sverdrup.discovery_year)
```

### Wellbores

```python
well = fp.well("31/2-1")          # By name
well = fp.wellbore("31/2-1")      # Alias for well()
well = fp.well(1234567)           # By npdid

print(well)
print(well.name)
print(well.total_depth)
print(well.purpose)
print(well.status)
```

### Facilities

```python
platform = fp.facility("TROLL A")
print(platform)
print(platform.kind)
print(platform.phase)
print(platform.water_depth)
```

### Pipelines

```python
pipe = fp.pipeline("STATPIPE")
print(pipe)
print(pipe.medium)
print(pipe.from_facility)
print(pipe.to_facility)
```

### Licenses

```python
license = fp.license("PL001")
print(license)
print(license.status)
print(license.granted_date)
```

### Companies

```python
equinor = fp.company("equinor")
print(equinor)
print(equinor.short_name)
print(equinor.org_number)
```

### Additional Entity Types

```python
# Plays (geological)
play = fp.play("UPPER JURASSIC")

# Blocks
block = fp.block("34/10")

# Quadrants
quad = fp.quadrant("34")

# Onshore facilities (TUF)
tuf = fp.tuf("KOLLSNES")

# Seismic surveys
survey = fp.seismic("NPD-1901")

# Stratigraphy
formation = fp.stratigraphy("DRAUPNE")

# Business arrangements
ba = fp.business_arrangement("TROLL UNIT")
```

---

## Exploring Data

### Entity Display

Each entity has a formatted `print()` output:

```python
troll = fp.field("troll")
print(troll)
```

Output:

```text
FIELD: TROLL
============================================================
Status:     Producing              Area:      North Sea
HC Type:    OIL/GAS               Discovered: 1979
Operator:   Equinor Energy AS

Partners (PL054):
  Equinor Energy AS                    30.58%  (operator)
  Petoro AS                            56.00%
  ...

Explore: .reserves  .wells  .licensees  .operators
```

### Related Tables

Access related data directly as DataFrames:

```python
troll = fp.field("troll")

# Direct attribute access for related tables
troll.field_reserves        # DataFrame of reserves
troll.field_licensee_hst    # DataFrame of licensee history
troll.field_operator_hst    # DataFrame of operator history

# Generic related() method
troll.related('field_reserves')
troll.related('discovery')  # Related discoveries
```

### Exploring Connections

```python
troll = fp.field("troll")

# Get list of connected tables
connections = troll.connections
print(connections['incoming'])  # Tables that reference this field
# ['field_reserves', 'field_licensee_hst', 'field_operator_hst', ...]

print(connections['outgoing'])  # Tables this field references
# ['company', 'licence', ...]

# Get actual filtered data for all connections
full_conns = troll.full_connections
reserves_df = full_conns['incoming']['field_reserves']
operator_df = full_conns['outgoing']['company']
```

### Partners and Ownership

```python
troll = fp.field("troll")

# Get partners with ownership info
partners = troll.partners
print(partners)
# Partners (5):
# ============================================================
# Company                                   Share %  Operator
# ------------------------------------------------------------
# Equinor Energy AS                           30.58  *
# Petoro AS                                   56.00
# ...

# Iterate partners (list of dicts)
for partner in partners:
    print(f"{partner['company']}: {partner['share']}%")
```

---

## Raw DataFrame Access

### Direct Table Access

```python
# Get any table as DataFrame
fields_df = fp.db.get('field')
discoveries_df = fp.db.get('discovery')
wellbores_df = fp.db.get('wellbore')

# Shorthand via fp.df() (auto-syncs if auto_sync=True)
fields_df = fp.df('field')

# Safe access (returns None if not exists)
df = fp.db.get_or_none('field_reserves')
```

### Convenience Methods

```python
# Get all fields
fields = fp.fields()

# Filter by status
producing = fp.fields(status='Producing')

# Get all discoveries
discoveries = fp.discoveries()

# Filter by year
discoveries_2023 = fp.discoveries(year=2023)

# Get all wellbores
wells = fp.wells()
```

### List Available Tables

```python
# Tables downloaded locally
fp.list_tables()
# ['block', 'company', 'discovery', 'facility', 'field', ...]

# Filter by prefix
fp.list_tables('field')
# ['field', 'field_reserves', 'field_licensee_hst', ...]

# Tables available on API
fp.api_tables()
# ['block', 'business_arrangement_area', 'company', ...]

# Filter API tables
fp.api_tables('wellbore')
# ['wellbore', 'wellbore_casing', 'wellbore_core', ...]
```

### Download Without Storing

```python
# Download data but don't store locally
df = fp.download('field')

# Download and store
df = fp.download('field', store=True)

# Download with filter
df = fp.download('wellbore', where="wlbStatus='COMPLETED'")
```

---

## Database Status

### Print Status

```python
fp.status()
```

Output:

```text
Database Status
================
Location: ./factpages_data/factpages.db
Tables: 53
Total records: 450,234

Top tables by size:
  wellbore_formation_top    125,432 records
  wellbore                   9,731 records
  licence_licensee_hst       8,234 records
  ...
```

### Detailed Database Info

```python
# List all tables
tables = fp.db.list_datasets()

# Get record count for a table
count = fp.db.get_record_count('wellbore')

# Check if table exists
exists = fp.db.has_dataset('field')

# Get last sync time
last_sync = fp.db.get_last_sync('field')
```

---

## Graph Building

Export data for knowledge graph libraries like [rusty-graph](https://github.com/your/rusty-graph):

### Quick Export

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

### Step-by-Step Loading

```python
# Get all nodes with column renaming
all_nodes = fp.graph.all_nodes(rename=True)
for node_type, df in all_nodes.items():
    print(f"{node_type}: {len(df)} nodes")

# Get connection specifications
connectors = fp.graph.all_connectors()
print(f"Found {len(connectors)} connection types")

# Load specific entity types
field_nodes = fp.graph.nodes('field', rename=True)
wellbore_nodes = fp.graph.nodes('wellbore', rename=True)
```

---

## Configuration

### Custom Client Configuration

```python
from factpages_py import Factpages, ClientConfig

config = ClientConfig(
    timeout=60,                    # Request timeout (seconds)
    connect_timeout=10,            # Connection timeout
    max_retries=5,                 # Max retry attempts
    rate_limit=0.2,                # Min seconds between requests
    pool_connections=20,           # Connection pool size
)

fp = Factpages(config=config)
```

### Auto-Sync Mode

```python
# Auto-download missing datasets when accessed
fp = Factpages(auto_sync=True)

# Now this will auto-download 'field' if not present
troll = fp.field("troll")
```

---

## Available Datasets

The API provides access to 75+ datasets organized by category:

| Category | Examples | Description |
|----------|----------|-------------|
| **Core Entities** | field, discovery, wellbore, facility | Main petroleum entities |
| **Company** | company | Operator and licensee information |
| **Licensing** | licence, licence_licensee_hst | License data and history |
| **Field Details** | field_reserves, field_licensee_hst | Field-specific tables |
| **Discovery Details** | discovery_reserves, discovery_operator_hst | Discovery-specific tables |
| **Wellbore Details** | wellbore_core, wellbore_dst, wellbore_formation_top | Well data |
| **Facility Details** | facility_function, pipeline | Infrastructure |
| **Seismic** | seismic_acquisition | Survey data |
| **Stratigraphy** | strat_litho, strat_chrono | Geological formations |
| **Administrative** | block, quadrant | Geographic divisions |

```python
# See all available datasets
fp.api_tables()

# Get dataset categories
from factpages_py import LAYERS, TABLES
print(f"Layers (with geometry): {len(LAYERS)}")
print(f"Tables (no geometry): {len(TABLES)}")
```

---

## Examples

### Find All Producing Fields

```python
fp = Factpages()
fp.sync('field')

producing = fp.fields(status='Producing')
print(f"Found {len(producing)} producing fields")

for _, row in producing.iterrows():
    print(f"  {row['fldName']}: {row['cmpLongName']}")
```

### Analyze Wellbore Depths

```python
import pandas as pd

fp = Factpages()
fp.sync('wellbore')

wells = fp.wells()
print(f"Average depth: {wells['wlbTotalDepth'].mean():.0f}m")
print(f"Deepest well: {wells['wlbTotalDepth'].max():.0f}m")
```

### Export Field-Company Relationships

```python
fp = Factpages()
fp.sync(['field', 'field_licensee_hst', 'company'])

# Build relationship table
fields = fp.db.get('field')
licensees = fp.db.get('field_licensee_hst')
companies = fp.db.get('company')

# Merge for full picture
relationships = licensees.merge(
    fields[['fldNpdidField', 'fldName']],
    on='fldNpdidField'
).merge(
    companies[['cmpNpdidCompany', 'cmpLongName']],
    on='cmpNpdidCompany'
)

print(relationships[['fldName', 'cmpLongName', 'fldLicenseeDateValidFrom']])
```

---

## License

MIT License

## Acknowledgments

Data provided by the [Norwegian Offshore Directorate](https://www.sodir.no/) (Sodir).

## Links

- [Sodir FactPages](https://factpages.sodir.no/)
- [Sodir REST API Documentation](https://factmaps.sodir.no/api/rest/services/DataService/Data/FeatureServer)
- [GitHub Repository](https://github.com/your-username/factpages-py)
