"""
Local Database Management

High-performance local storage for Factpages data using SQLite.

Storage structure:
    factpages_data/
    ├── factpages.db         # SQLite database with all tables
    └── sideloaded.db        # Sideloaded data (separate file)

SQLite provides:
- Single file storage (no file clutter)
- Multiple tables with different schemas
- Fast queries with indexing
- Type preservation
- Standard Python library support
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# =============================================================================
# Dataset Organization
# =============================================================================

FILE_MAPPING = {
    "entities": [
        "discovery", "field", "wellbore", "facility", "pipeline",
        "licence", "play", "block", "quadrant", "company",
    ],
    "geometries": [
        "discovery", "field", "wellbore", "facility", "pipeline",
        "licence", "block", "quadrant", "play",
    ],
    "production": [
        "field_production_monthly", "field_production_yearly",
        "field_investment_yearly", "field_pipeline_transport",
        "field_facility_transport", "csd_injection",
    ],
    "supporting": [
        "strat_chrono", "strat_litho", "strat_litho_wellbore", "strat_litho_wellbore_core",
        "discovery_reserves", "discovery_resource", "discovery_operator",
        "discovery_area", "discovery_resource_chrono", "discovery_resource_litho",
        "field_reserves", "field_description", "field_activity_status",
        "field_area", "field_operator", "field_owner", "field_licensee",
        "field_in_place_volumes",
        "wellbore_casing", "wellbore_core_photo", "wellbore_dst", "wellbore_history",
        "wellbore_chrono_strat", "wellbore_litho_strat", "wellbore_coordinates",
        "wellbore_document", "wellbore_log", "wellbore_mud", "wellbore_oil_sample",
        "wellbore_exploration_all", "wellbore_development_all",
        "facility_history", "tuf", "tuf_owner", "tuf_operator",
        "licence_licensee_history", "licence_operator_history",
        "licence_transfer_history", "licence_phase_history",
        "licence_task", "licence_document", "licence_additional_area",
        "licensing_activity",
        "business_arrangement_area", "business_arrangement_operator",
        "business_arrangement_licensee_history", "business_arrangement_transfer_history",
        "seismic_acquisition", "seismic_acquisition_area", "seismic_acquisition_progress",
    ],
}


def get_category_for_dataset(dataset: str) -> str:
    """Determine which category a dataset belongs to."""
    for category, datasets in FILE_MAPPING.items():
        if dataset in datasets:
            return category
    return "supporting"


# =============================================================================
# Database Class
# =============================================================================

class Database:
    """
    High-performance local database using SQLite storage.

    All datasets are stored in a single SQLite database file for:
    - Clean single-file storage
    - Fast queries with SQL
    - Multiple tables with different schemas
    - Built-in Python support

    Example:
        >>> db = Database()  # Uses ./factpages_data by default
        >>>
        >>> # Check if dataset is available locally
        >>> if db.has_dataset('discovery'):
        ...     discoveries = db.get('discovery')
        >>>
        >>> # Get sync status
        >>> db.print_status()
    """

    DB_FILE = "factpages.db"
    SIDELOAD_DB_FILE = "sideloaded.db"
    SIDELOAD_PREFIX = "sideload_"

    def __init__(self, data_dir: Union[str, Path] = "./factpages_data"):
        """
        Initialize the database.

        Args:
            data_dir: Directory to store the database files (default: ./factpages_data)

        Storage structure:
            factpages_data/
            ├── factpages.db      # Main database (all API tables)
            └── sideloaded.db     # Sideloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / self.DB_FILE
        self.sideload_db_path = self.data_dir / self.SIDELOAD_DB_FILE

        # In-memory cache for frequently accessed data
        self._cache: dict[str, pd.DataFrame] = {}

        # Column name cache for efficient schema lookups
        self._columns_cache: dict[str, list[str]] = {}

        # Connection pool for persistent connections
        self._connections: dict[Path, sqlite3.Connection] = {}

        # Initialize databases
        self._init_db(self.db_path)
        self._init_db(self.sideload_db_path)

    def _init_db(self, db_path: Path) -> None:
        """Initialize database with metadata table."""
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _metadata (
                    dataset TEXT PRIMARY KEY,
                    last_sync TEXT,
                    record_count INTEGER,
                    source TEXT,
                    checksum TEXT,
                    category TEXT
                )
            """)
            # Template customization storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _templates (
                    entity_type TEXT PRIMARY KEY,
                    template TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()

    def _get_db_path(self, dataset: str) -> Path:
        """Get the database path for a dataset."""
        if dataset.startswith(self.SIDELOAD_PREFIX):
            return self.sideload_db_path
        return self.db_path

    def _is_sideloaded(self, dataset: str) -> bool:
        """Check if a dataset name indicates sideloaded data."""
        return dataset.startswith(self.SIDELOAD_PREFIX)

    def _get_connection(self, db_path: Path) -> sqlite3.Connection:
        """
        Get or create a persistent connection with optimized settings.

        Uses WAL mode for better concurrency and read performance.
        """
        if db_path not in self._connections:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            # Enable Write-Ahead Logging for better read concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Optimize for read-heavy workloads
            conn.execute("PRAGMA synchronous=NORMAL")
            # 64MB cache for better performance
            conn.execute("PRAGMA cache_size=-64000")
            # Memory-map for faster reads (256MB)
            conn.execute("PRAGMA mmap_size=268435456")
            self._connections[db_path] = conn
        return self._connections[db_path]

    def close(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()
        self._cache.clear()
        self._columns_cache.clear()

    def __del__(self):
        """Cleanup connections on garbage collection."""
        self.close()

    # =========================================================================
    # Core Operations
    # =========================================================================

    def has_dataset(self, dataset: str) -> bool:
        """Check if a dataset exists locally."""
        db_path = self._get_db_path(dataset)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (dataset,)
            )
            return cursor.fetchone() is not None

    def get(self, dataset: str) -> pd.DataFrame:
        """
        Get a dataset as a DataFrame.

        Uses memory cache for repeated access.

        Args:
            dataset: Dataset name

        Returns:
            DataFrame with the dataset records

        Raises:
            KeyError: If dataset not found locally
        """
        # Check cache first
        if dataset in self._cache:
            return self._cache[dataset]

        if not self.has_dataset(dataset):
            raise KeyError(
                f"Dataset '{dataset}' not found locally. "
                f"Use sync() to download it first."
            )

        db_path = self._get_db_path(dataset)
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{dataset}"', conn)

        # Cache if not too large (< 100MB estimated)
        if len(df) < 500_000:
            self._cache[dataset] = df

        return df

    def get_or_none(self, dataset: str) -> Optional[pd.DataFrame]:
        """Get a dataset, returning None if not found."""
        try:
            return self.get(dataset)
        except KeyError:
            return None

    def query(
        self,
        dataset: str,
        where: Optional[dict] = None,
        columns: Optional[list[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query a dataset with SQL-level filtering (much faster than pandas filtering).

        Args:
            dataset: Dataset name
            where: Dict of {column: value} for WHERE clause (uses AND for multiple)
            columns: List of columns to return (None = all columns)
            limit: Maximum rows to return

        Returns:
            Filtered DataFrame

        Example:
            >>> db.query('field_reserves', where={'fldNpdidField': 43506})
            >>> db.query('profiles', where={'fldNpdidField': 43506}, limit=100)
        """
        if not self.has_dataset(dataset):
            return pd.DataFrame()

        # Build query
        cols = "*" if columns is None else ", ".join(f'"{c}"' for c in columns)
        query = f'SELECT {cols} FROM "{dataset}"'
        params = []

        if where:
            conditions = []
            for col, val in where.items():
                conditions.append(f'"{col}" = ?')
                params.append(val)
            query += " WHERE " + " AND ".join(conditions)

        if limit:
            query += f" LIMIT {int(limit)}"

        db_path = self._get_db_path(dataset)
        conn = self._get_connection(db_path)
        return pd.read_sql_query(query, conn, params=params if params else None)

    def query_exists(self, dataset: str, column: str, value) -> bool:
        """
        Check if a value exists in a column (fast indexed lookup).

        Args:
            dataset: Dataset name
            column: Column to check
            value: Value to find

        Returns:
            True if value exists
        """
        if not self.has_dataset(dataset):
            return False

        db_path = self._get_db_path(dataset)
        conn = self._get_connection(db_path)
        cursor = conn.execute(
            f'SELECT 1 FROM "{dataset}" WHERE "{column}" = ? LIMIT 1',
            (value,)
        )
        return cursor.fetchone() is not None

    def get_unique_values(self, dataset: str, column: str) -> set:
        """
        Get unique values from a column using SQL DISTINCT (much faster than pandas).

        Args:
            dataset: Dataset name
            column: Column to get unique values from

        Returns:
            Set of unique values
        """
        if not self.has_dataset(dataset):
            return set()

        db_path = self._get_db_path(dataset)
        conn = self._get_connection(db_path)
        cursor = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{dataset}" WHERE "{column}" IS NOT NULL'
        )
        return {row[0] for row in cursor.fetchall()}

    def get_columns(self, dataset: str) -> Optional[list[str]]:
        """
        Get column names for a dataset without loading full data.

        Uses SQLite PRAGMA for efficient schema lookup.

        Args:
            dataset: Dataset name

        Returns:
            List of column names, or None if dataset not found
        """
        # Check column cache first
        if dataset in self._columns_cache:
            return self._columns_cache[dataset]

        # Check if in data cache (already loaded)
        if dataset in self._cache:
            cols = list(self._cache[dataset].columns)
            self._columns_cache[dataset] = cols
            return cols

        # Query SQLite schema directly (fast, no data loading)
        if not self.has_dataset(dataset):
            return None

        db_path = self._get_db_path(dataset)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(f'PRAGMA table_info("{dataset}")')
            cols = [row[1] for row in cursor.fetchall()]  # row[1] is column name

        self._columns_cache[dataset] = cols
        return cols

    def put(
        self,
        dataset: str,
        df: pd.DataFrame,
        source: str = "api",
        checksum: Optional[str] = None
    ) -> None:
        """
        Store a dataset in the local database.

        Args:
            dataset: Dataset name (prefix with 'sideload_' for sideloaded data)
            df: DataFrame with records
            source: Data source identifier
            checksum: Optional checksum for change detection
        """
        db_path = self._get_db_path(dataset)

        with sqlite3.connect(db_path) as conn:
            # Store the data (replace if exists)
            df.to_sql(dataset, conn, if_exists='replace', index=False)

            # Create indexes on ID columns for fast joins/lookups
            for col in df.columns:
                if 'Npdid' in col or col.endswith('Id'):
                    try:
                        index_name = f"idx_{dataset}_{col}"
                        conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{dataset}" ("{col}")')
                    except sqlite3.OperationalError:
                        pass  # Index might already exist or column type not indexable

            # Update metadata
            conn.execute("""
                INSERT OR REPLACE INTO _metadata
                (dataset, last_sync, record_count, source, checksum, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset,
                datetime.now().isoformat(),
                len(df),
                source,
                checksum,
                "sideloaded" if self._is_sideloaded(dataset) else get_category_for_dataset(dataset)
            ))
            conn.commit()

        # Update caches
        self._columns_cache[dataset] = list(df.columns)
        if len(df) < 500_000:
            self._cache[dataset] = df.copy()

    def delete(self, dataset: str) -> bool:
        """
        Remove a dataset from the local database.

        Returns:
            True if deleted, False if not found
        """
        if not self.has_dataset(dataset):
            return False

        db_path = self._get_db_path(dataset)
        with sqlite3.connect(db_path) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{dataset}"')
            conn.execute("DELETE FROM _metadata WHERE dataset = ?", (dataset,))
            conn.commit()

        # Clear from cache
        self._cache.pop(dataset, None)
        return True

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    # =========================================================================
    # Sync Tracking
    # =========================================================================

    def get_sync_info(self, dataset: str) -> Optional[dict]:
        """Get sync metadata for a dataset."""
        db_path = self._get_db_path(dataset)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM _metadata WHERE dataset = ?",
                (dataset,)
            )
            row = cursor.fetchone()
            if row:
                cols = [d[0] for d in cursor.description]
                return dict(zip(cols, row))
        return None

    def get_last_sync(self, dataset: str) -> Optional[datetime]:
        """Get the last sync time for a dataset."""
        info = self.get_sync_info(dataset)
        if info and info.get("last_sync"):
            return datetime.fromisoformat(info["last_sync"])
        return None

    def get_record_count(self, dataset: str) -> int:
        """Get the record count for a dataset."""
        info = self.get_sync_info(dataset)
        return info.get("record_count", 0) if info else 0

    def is_stale(self, dataset: str, max_age_days: int = 7) -> bool:
        """
        Check if a dataset is stale.

        Args:
            dataset: Dataset name
            max_age_days: Maximum age in days

        Returns:
            True if stale or not synced
        """
        last_sync = self.get_last_sync(dataset)
        if not last_sync:
            return True

        age = datetime.now() - last_sync
        return age.days >= max_age_days

    # =========================================================================
    # Status & Reporting
    # =========================================================================

    def _get_all_metadata(self, db_path: Path) -> dict:
        """Get all metadata from a database."""
        result = {}
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT * FROM _metadata")
            cols = [d[0] for d in cursor.description]
            for row in cursor:
                info = dict(zip(cols, row))
                result[info['dataset']] = info
        return result

    def _get_db_size(self, db_path: Path) -> int:
        """Get database file size in bytes."""
        if db_path.exists():
            return db_path.stat().st_size
        return 0

    def status(self, include_sideloaded: bool = True) -> dict:
        """Get status of all datasets."""
        datasets_info = self._get_all_metadata(self.db_path)

        if include_sideloaded and self.sideload_db_path.exists():
            datasets_info.update(self._get_all_metadata(self.sideload_db_path))

        # Group by category
        by_category: dict[str, list] = {
            "entities": [],
            "geometries": [],
            "production": [],
            "supporting": [],
            "sideloaded": [],
        }

        for name, info in datasets_info.items():
            category = info.get("category", "supporting")
            by_category.setdefault(category, []).append({
                "name": name,
                "record_count": info.get("record_count", 0),
                "last_sync": info.get("last_sync"),
            })

        main_size = self._get_db_size(self.db_path)
        sideload_size = self._get_db_size(self.sideload_db_path)

        return {
            "data_dir": str(self.data_dir.absolute()),
            "db_file": str(self.db_path),
            "sideload_db_file": str(self.sideload_db_path),
            "total_datasets": len(datasets_info),
            "total_size_mb": (main_size + sideload_size) / (1024 * 1024),
            "main_db_size_mb": main_size / (1024 * 1024),
            "sideload_db_size_mb": sideload_size / (1024 * 1024),
            "by_category": by_category,
        }

    def print_status(self) -> None:
        """Print a formatted status report."""
        status = self.status()

        print("\n" + "=" * 60)
        print("LOCAL DATABASE STATUS")
        print("=" * 60)
        print(f"Database:   {status['db_file']}")
        print(f"Size:       {status['main_db_size_mb']:.2f} MB")
        if status['sideload_db_size_mb'] > 0:
            print(f"Sideloaded: {status['sideload_db_file']}")
            print(f"            {status['sideload_db_size_mb']:.2f} MB")
        print()

        for category in ["entities", "geometries", "production", "supporting", "sideloaded"]:
            datasets = status["by_category"].get(category, [])
            if datasets:
                print(f"{category.upper()} ({len(datasets)} tables)")
                for ds in sorted(datasets, key=lambda x: x["name"]):
                    count = ds["record_count"]
                    print(f"  {ds['name']:<40} {count:>8,} records")
                print()

        print("-" * 60)
        print(f"Total: {status['total_datasets']} tables, {status['total_size_mb']:.2f} MB")

    def list_datasets(self, include_sideloaded: bool = True) -> list[str]:
        """List all datasets available locally."""
        datasets = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != '_metadata'"
            )
            datasets.extend(row[0] for row in cursor)

        if include_sideloaded and self.sideload_db_path.exists():
            with sqlite3.connect(self.sideload_db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name != '_metadata'"
                )
                datasets.extend(row[0] for row in cursor)

        return sorted(datasets)

    def list_sideloaded_datasets(self) -> list[str]:
        """List only sideloaded datasets."""
        if not self.sideload_db_path.exists():
            return []

        with sqlite3.connect(self.sideload_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != '_metadata'"
            )
            return sorted(row[0] for row in cursor)

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def get_category(self, category: str) -> dict[str, pd.DataFrame]:
        """
        Load all datasets in a category.

        Args:
            category: One of 'entities', 'geometries', 'production', 'supporting'

        Returns:
            Dict mapping dataset names to DataFrames
        """
        if category not in FILE_MAPPING:
            raise ValueError(f"Unknown category: {category}")

        result = {}
        for dataset in FILE_MAPPING[category]:
            if self.has_dataset(dataset):
                result[dataset] = self.get(dataset)

        return result

    def export_to_parquet(self, output_dir: Union[str, Path], datasets: Optional[list[str]] = None) -> None:
        """
        Export datasets to Parquet format.

        Args:
            output_dir: Output directory
            datasets: List of datasets to export (default: all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = datasets or self.list_datasets()

        for dataset in datasets:
            if self.has_dataset(dataset):
                df = self.get(dataset)
                output_path = output_dir / f"{dataset}.parquet"
                df.to_parquet(output_path, index=False, compression='snappy')
                print(f"  Exported {dataset}: {len(df)} records")

    def export_to_csv(self, output_dir: Union[str, Path], datasets: Optional[list[str]] = None) -> None:
        """
        Export datasets to CSV format for human readability.

        Args:
            output_dir: Output directory
            datasets: List of datasets to export (default: all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = datasets or self.list_datasets()

        for dataset in datasets:
            if self.has_dataset(dataset):
                df = self.get(dataset)
                output_path = output_dir / f"{dataset}.csv"
                df.to_csv(output_path, index=False)
                print(f"  Exported {dataset}: {len(df)} records")

    def export_to_json(self, output_dir: Union[str, Path]) -> None:
        """
        Export all datasets to JSON format (for portability).

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset in self.list_datasets():
            df = self.get(dataset)
            output_path = output_dir / f"{dataset}.json"
            df.to_json(output_path, orient='records', indent=2)
            print(f"  Exported {dataset}: {len(df)} records")

    def export_to_excel(
        self,
        output_path: Union[str, Path],
        category: Optional[str] = None,
        datasets: Optional[list[str]] = None
    ) -> None:
        """
        Export datasets to Excel workbook (one sheet per dataset).

        Args:
            output_path: Output Excel file path
            category: Export only this category
            datasets: Specific datasets to export
        """
        output_path = Path(output_path)

        if category and category in FILE_MAPPING:
            datasets = [d for d in FILE_MAPPING[category] if self.has_dataset(d)]
        elif datasets is None:
            datasets = self.list_datasets()

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for dataset in datasets:
                if self.has_dataset(dataset):
                    df = self.get(dataset)
                    # Excel sheet names limited to 31 chars
                    sheet_name = dataset[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  Added sheet: {dataset} ({len(df)} rows)")

        print(f"\nExported to: {output_path}")

    def import_from_parquet(
        self,
        parquet_path: Union[str, Path],
        dataset: Optional[str] = None,
        source: str = "parquet_import"
    ) -> pd.DataFrame:
        """
        Import a dataset from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            dataset: Dataset name (default: filename without extension)
            source: Source identifier for tracking

        Returns:
            Imported DataFrame
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if dataset is None:
            dataset = parquet_path.stem

        df = pd.read_parquet(parquet_path)
        self.put(dataset, df, source=source)
        print(f"  Imported {dataset}: {len(df)} records")
        return df

    def import_from_csv(
        self,
        csv_path: Union[str, Path],
        dataset: Optional[str] = None,
        source: str = "csv_import",
        **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        Import a dataset from a CSV file.

        Args:
            csv_path: Path to CSV file
            dataset: Dataset name (default: filename without extension)
            source: Source identifier for tracking
            **read_csv_kwargs: Additional arguments for pd.read_csv

        Returns:
            Imported DataFrame
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if dataset is None:
            dataset = csv_path.stem

        df = pd.read_csv(csv_path, **read_csv_kwargs)
        self.put(dataset, df, source=source)
        print(f"  Imported {dataset}: {len(df)} records")
        return df

    def import_from_json(self, json_path: Union[str, Path], dataset: str) -> pd.DataFrame:
        """
        Import a dataset from JSON file.

        Args:
            json_path: Path to JSON file
            dataset: Dataset name to store as

        Returns:
            Imported DataFrame
        """
        df = pd.read_json(json_path)
        self.put(dataset, df, source="json_import")
        return df

    def import_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.parquet",
        source: str = "directory_import"
    ) -> dict[str, int]:
        """
        Import all matching files from a directory.

        Args:
            directory: Directory containing data files
            pattern: Glob pattern for files (*.parquet, *.csv, *.json)
            source: Source identifier for tracking

        Returns:
            Dict mapping dataset names to record counts
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results = {}
        files = sorted(directory.glob(pattern))

        if not files:
            print(f"No files matching '{pattern}' in {directory}")
            return results

        print(f"Importing {len(files)} files from {directory}")

        for file_path in files:
            dataset = file_path.stem

            try:
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix == '.json':
                    df = pd.read_json(file_path)
                else:
                    print(f"  Skipping unsupported format: {file_path.name}")
                    continue

                self.put(dataset, df, source=source)
                results[dataset] = len(df)
                print(f"  {dataset}: {len(df)} records")

            except Exception as e:
                print(f"  ERROR {dataset}: {e}")
                results[dataset] = -1

        print(f"\nImported {len([v for v in results.values() if v > 0])} datasets")
        return results

    # =========================================================================
    # Migration from Parquet
    # =========================================================================

    def migrate_from_parquet(self, parquet_dir: Optional[Union[str, Path]] = None) -> int:
        """
        Migrate existing parquet files to SQLite database.

        Args:
            parquet_dir: Directory with parquet files (default: self.data_dir)

        Returns:
            Number of tables migrated
        """
        parquet_dir = Path(parquet_dir) if parquet_dir else self.data_dir

        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            print("No parquet files found to migrate")
            return 0

        print(f"Migrating {len(parquet_files)} parquet files to SQLite...")

        migrated = 0
        for pq_file in parquet_files:
            dataset = pq_file.stem
            if dataset.startswith('_'):
                continue

            try:
                df = pd.read_parquet(pq_file)
                self.put(dataset, df, source="parquet_migration")
                print(f"  Migrated {dataset}: {len(df)} records")
                migrated += 1
            except Exception as e:
                print(f"  ERROR migrating {dataset}: {e}")

        print(f"\nMigrated {migrated} tables to SQLite")
        return migrated

    def cleanup_parquet_files(self, parquet_dir: Optional[Union[str, Path]] = None) -> int:
        """
        Remove old parquet files after migration.

        Args:
            parquet_dir: Directory with parquet files (default: self.data_dir)

        Returns:
            Number of files removed
        """
        parquet_dir = Path(parquet_dir) if parquet_dir else self.data_dir

        removed = 0
        for pq_file in parquet_dir.glob("*.parquet"):
            pq_file.unlink()
            removed += 1
            print(f"  Removed {pq_file.name}")

        # Also remove old metadata json
        old_metadata = parquet_dir / "_metadata.json"
        if old_metadata.exists():
            old_metadata.unlink()
            print("  Removed _metadata.json")

        # Remove sideloaded directory if empty
        sideload_dir = parquet_dir / "sideloaded"
        if sideload_dir.exists():
            for pq_file in sideload_dir.glob("*.parquet"):
                pq_file.unlink()
                removed += 1
            old_meta = sideload_dir / "_metadata.json"
            if old_meta.exists():
                old_meta.unlink()
            try:
                sideload_dir.rmdir()
            except OSError:
                pass

        print(f"\nRemoved {removed} parquet files")
        return removed

    # =========================================================================
    # Schema Versioning
    # =========================================================================

    SCHEMA_VERSION = 2  # Version 2 = SQLite storage

    def get_schema_version(self) -> int:
        """Get the current database schema version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_version'"
            )
            if cursor.fetchone() is None:
                return 1  # Old parquet-based schema

            cursor = conn.execute("SELECT version FROM _schema_version")
            row = cursor.fetchone()
            return row[0] if row else 1

    def set_schema_version(self, version: int) -> None:
        """Set the database schema version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER)
            """)
            conn.execute("DELETE FROM _schema_version")
            conn.execute("INSERT INTO _schema_version VALUES (?)", (version,))
            conn.commit()

    def needs_migration(self) -> bool:
        """Check if database needs migration to current schema."""
        # Check for old parquet files
        parquet_files = list(self.data_dir.glob("*.parquet"))
        return len(parquet_files) > 0

    def migrate(self) -> None:
        """
        Run database migrations to bring schema up to date.

        Migrates from parquet files to SQLite if needed.
        """
        if not self.needs_migration():
            print("Database is up to date (SQLite format)")
            return

        print("Migrating from parquet to SQLite...")
        migrated = self.migrate_from_parquet()

        if migrated > 0:
            self.set_schema_version(self.SCHEMA_VERSION)
            print("\nMigration complete!")
            print("Run db.cleanup_parquet_files() to remove old parquet files")

    def validate_integrity(self) -> dict:
        """
        Validate database integrity.

        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "datasets_checked": 0,
        }

        datasets = self.list_datasets()

        for dataset in datasets:
            try:
                df = self.get(dataset)
                info = self.get_sync_info(dataset)

                if info:
                    expected_count = info.get("record_count", 0)
                    actual_count = len(df)

                    if expected_count != actual_count:
                        results["warnings"].append(
                            f"{dataset}: record count mismatch "
                            f"(metadata: {expected_count}, actual: {actual_count})"
                        )

                results["datasets_checked"] += 1

            except Exception as e:
                results["errors"].append(f"Cannot read {dataset}: {e}")
                results["valid"] = False

        return results

    def vacuum(self) -> None:
        """
        Optimize database by reclaiming unused space.

        Run this after deleting many datasets.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
        print(f"Vacuumed {self.db_path.name}")

        if self.sideload_db_path.exists():
            with sqlite3.connect(self.sideload_db_path) as conn:
                conn.execute("VACUUM")
            print(f"Vacuumed {self.sideload_db_path.name}")

    def optimize(self, verbose: bool = True) -> dict:
        """
        Optimize database by creating indexes on ID columns and running ANALYZE.

        Creates indexes on columns containing 'Npdid' or ending with 'Id' for
        faster joins and lookups. Also updates SQLite query planner statistics.

        Args:
            verbose: Print progress messages

        Returns:
            Dict with optimization statistics

        Example:
            >>> fp.db.optimize()
            Creating indexes on 75 tables...
            Created 234 indexes
            Running ANALYZE...
            Done!
        """
        stats = {'indexes_created': 0, 'tables_processed': 0}

        tables = self.list_datasets(include_sideloaded=False)
        if verbose:
            print(f"Creating indexes on {len(tables)} tables...")

        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                if table.startswith('_'):
                    continue

                # Get columns for this table
                cols = self.get_columns(table)
                if not cols:
                    continue

                stats['tables_processed'] += 1

                # Create indexes on ID columns
                for col in cols:
                    if 'Npdid' in col or col.endswith('Id'):
                        try:
                            index_name = f"idx_{table}_{col}"
                            conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ("{col}")')
                            stats['indexes_created'] += 1
                        except sqlite3.OperationalError:
                            pass

            # Update query planner statistics
            if verbose:
                print(f"Created {stats['indexes_created']} indexes")
                print("Running ANALYZE...")
            conn.execute("ANALYZE")
            conn.commit()

        if verbose:
            print("Done!")

        return stats

    # =========================================================================
    # Template Customization
    # =========================================================================

    def save_template(self, entity_type: str, template: str) -> None:
        """
        Save a custom template for an entity type.

        Args:
            entity_type: Entity type (e.g., 'field', 'discovery')
            template: The template string (newline-separated lines)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO _templates (entity_type, template, updated_at)
                VALUES (?, ?, ?)
            """, (entity_type, template, datetime.now().isoformat()))
            conn.commit()

    def get_template(self, entity_type: str) -> Optional[str]:
        """
        Get custom template for an entity type.

        Args:
            entity_type: Entity type (e.g., 'field', 'discovery')

        Returns:
            Custom template string or None if no custom template exists
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT template FROM _templates WHERE entity_type = ?",
                (entity_type,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def delete_template(self, entity_type: str) -> bool:
        """
        Delete custom template, reverting to default.

        Args:
            entity_type: Entity type (e.g., 'field', 'discovery')

        Returns:
            True if template was deleted, False if no custom template existed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM _templates WHERE entity_type = ?",
                (entity_type,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_custom_templates(self) -> list[str]:
        """
        List all entity types that have custom templates.

        Returns:
            List of entity type names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT entity_type FROM _templates")
            return [row[0] for row in cursor.fetchall()]

    # =========================================================================
    # API Metadata (table and column descriptions)
    # =========================================================================

    def _init_metadata_tables(self) -> None:
        """Initialize API metadata tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Table descriptions from Metadata FeatureServer
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _table_descriptions (
                    table_name TEXT PRIMARY KEY,
                    table_id TEXT,
                    description TEXT,
                    fetched_at TEXT
                )
            """)
            # Column/attribute descriptions from Metadata FeatureServer
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _column_descriptions (
                    table_name TEXT,
                    column_name TEXT,
                    display_text TEXT,
                    data_type TEXT,
                    description TEXT,
                    PRIMARY KEY (table_name, column_name)
                )
            """)
            conn.commit()

    def save_table_descriptions(self, descriptions: list[dict]) -> int:
        """
        Save table descriptions from Metadata API.

        Args:
            descriptions: List of dicts with keys: id, TableName, Description

        Returns:
            Number of descriptions saved
        """
        self._init_metadata_tables()
        fetched_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for desc in descriptions:
                conn.execute("""
                    INSERT OR REPLACE INTO _table_descriptions
                    (table_name, table_id, description, fetched_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    desc.get('TableName', '').lower(),
                    str(desc.get('id', '')),
                    desc.get('Description', ''),
                    fetched_at
                ))
            conn.commit()

        return len(descriptions)

    def save_column_descriptions(self, descriptions: list[dict]) -> int:
        """
        Save column descriptions from Metadata API.

        Args:
            descriptions: List of dicts with keys: TableName, Tag, DisplayText, DataType, Description

        Returns:
            Number of descriptions saved
        """
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            for desc in descriptions:
                conn.execute("""
                    INSERT OR REPLACE INTO _column_descriptions
                    (table_name, column_name, display_text, data_type, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    desc.get('TableName', '').lower(),
                    desc.get('Tag', ''),
                    desc.get('DisplayText', ''),
                    desc.get('DataType', ''),
                    desc.get('Description', '')
                ))
            conn.commit()

        return len(descriptions)

    def get_table_description(self, table_name: str) -> Optional[str]:
        """
        Get the description for a table.

        Args:
            table_name: Table name (case-insensitive)

        Returns:
            Description string or None if not found
        """
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT description FROM _table_descriptions WHERE table_name = ?",
                (table_name.lower(),)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_column_description(self, table_name: str, column_name: str) -> Optional[dict]:
        """
        Get description for a specific column.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            Dict with display_text, data_type, description or None
        """
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT display_text, data_type, description
                FROM _column_descriptions
                WHERE table_name = ? AND column_name = ?
            """, (table_name.lower(), column_name))
            row = cursor.fetchone()
            if row:
                return {
                    'display_text': row[0],
                    'data_type': row[1],
                    'description': row[2]
                }
            return None

    def get_all_column_descriptions(self, table_name: str) -> dict[str, dict]:
        """
        Get all column descriptions for a table.

        Args:
            table_name: Table name

        Returns:
            Dict mapping column names to their description info
        """
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT column_name, display_text, data_type, description
                FROM _column_descriptions
                WHERE table_name = ?
            """, (table_name.lower(),))

            result = {}
            for row in cursor:
                result[row[0]] = {
                    'display_text': row[1],
                    'data_type': row[2],
                    'description': row[3]
                }
            return result

    def has_metadata(self) -> bool:
        """Check if API metadata has been fetched."""
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM _table_descriptions")
            count = cursor.fetchone()[0]
            return count > 0

    def get_metadata_age_days(self) -> Optional[int]:
        """Get age of metadata in days, or None if not fetched."""
        self._init_metadata_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT fetched_at FROM _table_descriptions LIMIT 1"
            )
            row = cursor.fetchone()
            if row and row[0]:
                fetched = datetime.fromisoformat(row[0])
                return (datetime.now() - fetched).days
            return None

    # =========================================================================
    # Remote Record Counts (expected counts from API)
    # =========================================================================

    def _init_remote_counts_table(self) -> None:
        """Initialize remote counts table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _remote_counts (
                    table_name TEXT PRIMARY KEY,
                    count INTEGER,
                    fetched_at TEXT
                )
            """)
            conn.commit()

    def save_remote_counts(self, counts: dict[str, int]) -> int:
        """
        Save remote record counts from API.

        Args:
            counts: Dict mapping table names to their remote counts

        Returns:
            Number of counts saved
        """
        self._init_remote_counts_table()
        fetched_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for table_name, count in counts.items():
                conn.execute("""
                    INSERT OR REPLACE INTO _remote_counts
                    (table_name, count, fetched_at)
                    VALUES (?, ?, ?)
                """, (table_name.lower(), count, fetched_at))
            conn.commit()

        return len(counts)

    def get_remote_count(self, table_name: str) -> Optional[int]:
        """
        Get expected remote count for a table.

        Args:
            table_name: Table name (case-insensitive)

        Returns:
            Expected count or None if not fetched
        """
        self._init_remote_counts_table()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT count FROM _remote_counts WHERE table_name = ?",
                (table_name.lower(),)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_remote_counts(self) -> dict[str, int]:
        """
        Get all remote counts.

        Returns:
            Dict mapping table names to their expected counts
        """
        self._init_remote_counts_table()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT table_name, count FROM _remote_counts")
            return {row[0]: row[1] for row in cursor}

    def get_remote_counts_age_days(self) -> Optional[int]:
        """Get age of remote counts in days, or None if not fetched."""
        self._init_remote_counts_table()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT fetched_at FROM _remote_counts LIMIT 1"
            )
            row = cursor.fetchone()
            if row and row[0]:
                fetched = datetime.fromisoformat(row[0])
                return (datetime.now() - fetched).days
            return None

    def has_remote_counts(self) -> bool:
        """Check if remote counts have been fetched."""
        self._init_remote_counts_table()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM _remote_counts")
            count = cursor.fetchone()[0]
            return count > 0
