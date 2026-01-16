"""
Local Database Management

High-performance local storage for Factpages data using Parquet format.

Storage structure:
    data_dir/
    ├── _metadata.json       # Sync timestamps, record counts
    ├── discovery.parquet    # Each dataset as separate Parquet file
    ├── field.parquet
    ├── wellbore.parquet
    └── ...

Parquet provides:
- Columnar storage optimized for analytical queries
- Excellent compression (typically 5-10x smaller than JSON)
- Type preservation (dates, integers, floats)
- Fast I/O with memory mapping
"""

import json
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
    High-performance local database using Parquet storage.

    Each dataset is stored as a separate Parquet file for:
    - Fast loading of individual datasets
    - Efficient compression
    - Type preservation
    - Memory-mapped reading

    Example:
        >>> db = Database('./data')
        >>>
        >>> # Check if dataset is available locally
        >>> if db.has_dataset('discovery'):
        ...     discoveries = db.get('discovery')
        >>>
        >>> # Get sync status
        >>> db.print_status()
    """

    METADATA_FILE = "_metadata.json"
    SIDELOAD_PREFIX = "sideload_"  # Prefix for sideloaded datasets
    SIDELOAD_DIR = "sideloaded"    # Subdirectory for sideloaded data

    def __init__(self, data_dir: Union[str, Path] = "./data"):
        """
        Initialize the database.

        Args:
            data_dir: Directory to store the Parquet files

        Storage structure:
            data_dir/
            ├── _metadata.json           # API data metadata
            ├── field.parquet            # API data
            ├── discovery.parquet
            └── sideloaded/              # Sideloaded data (separate!)
                ├── _metadata.json       # Sideloaded data metadata
                ├── projects.parquet
                └── tasks.parquet
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create sideloaded subdirectory
        self.sideload_dir = self.data_dir / self.SIDELOAD_DIR
        self.sideload_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed data
        self._cache: dict[str, pd.DataFrame] = {}
        self._metadata: Optional[dict] = None
        self._sideload_metadata: Optional[dict] = None

    # =========================================================================
    # Metadata Management
    # =========================================================================

    def _is_sideloaded(self, dataset: str) -> bool:
        """Check if a dataset name indicates sideloaded data."""
        return dataset.startswith(self.SIDELOAD_PREFIX)

    def _metadata_path(self, sideloaded: bool = False) -> Path:
        if sideloaded:
            return self.sideload_dir / self.METADATA_FILE
        return self.data_dir / self.METADATA_FILE

    def _load_metadata(self, sideloaded: bool = False) -> dict:
        """Load metadata from disk."""
        if sideloaded:
            if self._sideload_metadata is not None:
                return self._sideload_metadata
            cache_attr = '_sideload_metadata'
        else:
            if self._metadata is not None:
                return self._metadata
            cache_attr = '_metadata'

        path = self._metadata_path(sideloaded)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "version": "2.0",
                "format": "parquet",
                "type": "sideloaded" if sideloaded else "api",
                "created": datetime.now().isoformat(),
                "datasets": {},
            }

        setattr(self, cache_attr, metadata)
        return metadata

    def _save_metadata(self, sideloaded: bool = False) -> None:
        """Save metadata to disk."""
        if sideloaded:
            metadata = self._sideload_metadata
        else:
            metadata = self._metadata

        if metadata is None:
            return

        metadata["last_modified"] = datetime.now().isoformat()

        with open(self._metadata_path(sideloaded), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    # =========================================================================
    # File Operations
    # =========================================================================

    def _dataset_path(self, dataset: str) -> Path:
        """Get the Parquet file path for a dataset."""
        if self._is_sideloaded(dataset):
            # Sideloaded data goes in subdirectory
            return self.sideload_dir / f"{dataset}.parquet"
        return self.data_dir / f"{dataset}.parquet"

    def has_dataset(self, dataset: str) -> bool:
        """Check if a dataset exists locally."""
        return self._dataset_path(dataset).exists()

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

        path = self._dataset_path(dataset)
        if not path.exists():
            raise KeyError(
                f"Dataset '{dataset}' not found locally. "
                f"Use sync() to download it first."
            )

        # Load from Parquet
        df = pd.read_parquet(path)

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

        Note:
            Datasets prefixed with 'sideload_' are stored in a separate
            subdirectory to prevent mixing with API data.
        """
        is_sideloaded = self._is_sideloaded(dataset)
        path = self._dataset_path(dataset)

        # Save as Parquet with compression
        df.to_parquet(path, index=False, compression='snappy')

        # Update metadata (use correct metadata file based on data type)
        metadata = self._load_metadata(sideloaded=is_sideloaded)
        metadata["datasets"][dataset] = {
            "last_sync": datetime.now().isoformat(),
            "record_count": len(df),
            "source": source,
            "checksum": checksum,
            "category": "sideloaded" if is_sideloaded else get_category_for_dataset(dataset),
            "file_size_bytes": path.stat().st_size,
        }
        self._save_metadata(sideloaded=is_sideloaded)

        # Update cache
        if len(df) < 500_000:
            self._cache[dataset] = df.copy()

    def delete(self, dataset: str) -> bool:
        """
        Remove a dataset from the local database.

        Returns:
            True if deleted, False if not found
        """
        is_sideloaded = self._is_sideloaded(dataset)
        path = self._dataset_path(dataset)

        if path.exists():
            path.unlink()

            # Update metadata (use correct metadata file)
            metadata = self._load_metadata(sideloaded=is_sideloaded)
            if dataset in metadata.get("datasets", {}):
                del metadata["datasets"][dataset]
                self._save_metadata(sideloaded=is_sideloaded)

            # Clear from cache
            self._cache.pop(dataset, None)
            return True

        return False

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._metadata = None
        self._sideload_metadata = None

    # =========================================================================
    # Sync Tracking
    # =========================================================================

    def get_sync_info(self, dataset: str) -> Optional[dict]:
        """Get sync metadata for a dataset."""
        is_sideloaded = self._is_sideloaded(dataset)
        metadata = self._load_metadata(sideloaded=is_sideloaded)
        return metadata.get("datasets", {}).get(dataset)

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

    def status(self, include_sideloaded: bool = True) -> dict:
        """Get status of all datasets."""
        metadata = self._load_metadata()
        datasets_info = metadata.get("datasets", {}).copy()

        # Also include sideloaded data if requested
        if include_sideloaded:
            sideload_metadata = self._load_metadata(sideloaded=True)
            datasets_info.update(sideload_metadata.get("datasets", {}))

        # Group by category
        by_category: dict[str, list] = {
            "entities": [],
            "geometries": [],
            "production": [],
            "supporting": [],
            "sideloaded": [],
        }

        total_size = 0
        for name, info in datasets_info.items():
            category = info.get("category", "supporting")
            size = info.get("file_size_bytes", 0)
            total_size += size

            by_category.setdefault(category, []).append({
                "name": name,
                "record_count": info.get("record_count", 0),
                "last_sync": info.get("last_sync"),
                "size_mb": size / (1024 * 1024),
            })

        return {
            "data_dir": str(self.data_dir.absolute()),
            "sideload_dir": str(self.sideload_dir.absolute()),
            "total_datasets": len(datasets_info),
            "total_size_mb": total_size / (1024 * 1024),
            "by_category": by_category,
        }

    def print_status(self) -> None:
        """Print a formatted status report."""
        status = self.status()

        print("\n" + "=" * 60)
        print("LOCAL DATABASE STATUS")
        print("=" * 60)
        print(f"API Data:       {status['data_dir']}")
        print(f"Sideloaded:     {status['sideload_dir']}")
        print(f"Format:         Parquet (compressed)")
        print()

        for category in ["entities", "geometries", "production", "supporting", "sideloaded"]:
            datasets = status["by_category"].get(category, [])
            if datasets:
                print(f"{category.upper()} ({len(datasets)} datasets)")
                for ds in sorted(datasets, key=lambda x: x["name"]):
                    count = ds["record_count"]
                    size = ds["size_mb"]
                    print(f"  {ds['name']:<35} {count:>8,} records  {size:>6.2f} MB")
                print()

        print("-" * 60)
        print(f"Total: {status['total_datasets']} datasets, {status['total_size_mb']:.2f} MB")

    def list_datasets(self, include_sideloaded: bool = True) -> list[str]:
        """List all datasets available locally."""
        metadata = self._load_metadata()
        datasets = list(metadata.get("datasets", {}).keys())

        if include_sideloaded:
            sideload_metadata = self._load_metadata(sideloaded=True)
            datasets.extend(sideload_metadata.get("datasets", {}).keys())

        return sorted(datasets)

    def list_sideloaded_datasets(self) -> list[str]:
        """List only sideloaded datasets."""
        metadata = self._load_metadata(sideloaded=True)
        return sorted(metadata.get("datasets", {}).keys())

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

        # Use filename as dataset name if not specified
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
    # Schema Versioning
    # =========================================================================

    SCHEMA_VERSION = 1

    def get_schema_version(self) -> int:
        """Get the current database schema version."""
        metadata = self._load_metadata()
        return metadata.get("schema_version", 1)

    def set_schema_version(self, version: int) -> None:
        """Set the database schema version."""
        metadata = self._load_metadata()
        metadata["schema_version"] = version
        self._save_metadata()

    def needs_migration(self) -> bool:
        """Check if database needs migration to current schema."""
        return self.get_schema_version() < self.SCHEMA_VERSION

    def migrate(self) -> None:
        """
        Run database migrations to bring schema up to date.

        Migrations are run in order from current version to SCHEMA_VERSION.
        """
        current = self.get_schema_version()
        target = self.SCHEMA_VERSION

        if current >= target:
            print(f"Database schema is up to date (v{current})")
            return

        print(f"Migrating database from v{current} to v{target}")

        # Run migrations in order
        migrations = {
            # version: migration_function
            # 2: self._migrate_v1_to_v2,
            # 3: self._migrate_v2_to_v3,
        }

        for version in range(current + 1, target + 1):
            if version in migrations:
                print(f"  Running migration to v{version}...")
                migrations[version]()

        self.set_schema_version(target)
        print(f"Migration complete. Database is now at v{target}")

    def validate_integrity(self) -> dict:
        """
        Validate database integrity.

        Checks:
        - All Parquet files are readable
        - Metadata matches actual files
        - No orphaned files

        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "datasets_checked": 0,
        }

        metadata = self._load_metadata()
        metadata_datasets = set(metadata.get("datasets", {}).keys())

        # Check files on disk
        parquet_files = set(
            f.stem for f in self.data_dir.glob("*.parquet")
        )

        # Datasets in metadata but missing files
        missing_files = metadata_datasets - parquet_files
        for dataset in missing_files:
            results["errors"].append(f"Missing file for dataset: {dataset}")
            results["valid"] = False

        # Files on disk not in metadata
        orphaned_files = parquet_files - metadata_datasets
        for dataset in orphaned_files:
            results["warnings"].append(f"Orphaned file (not in metadata): {dataset}.parquet")

        # Validate each dataset is readable
        for dataset in metadata_datasets & parquet_files:
            try:
                df = pd.read_parquet(self._dataset_path(dataset))
                expected_count = metadata["datasets"][dataset].get("record_count", 0)
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

    def repair(self, fix_orphans: bool = True, update_counts: bool = True) -> None:
        """
        Repair database inconsistencies.

        Args:
            fix_orphans: Add orphaned files to metadata
            update_counts: Update record counts in metadata
        """
        print("Repairing database...")

        metadata = self._load_metadata()
        metadata_datasets = set(metadata.get("datasets", {}).keys())
        parquet_files = set(f.stem for f in self.data_dir.glob("*.parquet"))

        # Fix orphaned files
        if fix_orphans:
            orphaned = parquet_files - metadata_datasets
            for dataset in orphaned:
                try:
                    df = pd.read_parquet(self._dataset_path(dataset))
                    metadata["datasets"][dataset] = {
                        "last_sync": None,
                        "record_count": len(df),
                        "source": "repair",
                        "category": get_category_for_dataset(dataset),
                        "file_size_bytes": self._dataset_path(dataset).stat().st_size,
                    }
                    print(f"  Added orphaned dataset to metadata: {dataset}")
                except Exception as e:
                    print(f"  Could not repair {dataset}: {e}")

        # Update record counts
        if update_counts:
            for dataset in metadata_datasets & parquet_files:
                try:
                    df = pd.read_parquet(self._dataset_path(dataset))
                    old_count = metadata["datasets"][dataset].get("record_count", 0)
                    new_count = len(df)

                    if old_count != new_count:
                        metadata["datasets"][dataset]["record_count"] = new_count
                        metadata["datasets"][dataset]["file_size_bytes"] = \
                            self._dataset_path(dataset).stat().st_size
                        print(f"  Updated {dataset} count: {old_count} -> {new_count}")

                except Exception as e:
                    print(f"  Could not update {dataset}: {e}")

        # Remove missing datasets from metadata
        missing = metadata_datasets - parquet_files
        for dataset in missing:
            del metadata["datasets"][dataset]
            print(f"  Removed missing dataset from metadata: {dataset}")

        self._save_metadata()
        print("Repair complete")
