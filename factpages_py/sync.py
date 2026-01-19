"""
Synchronization Engine

Handles synchronization between the remote API and local database.
Key goals:
- Minimize server traffic (fetch only once when possible)
- Smart change detection using record counts
- Configurable sync strategies
- Retry with exponential backoff
- Resume interrupted syncs
- Progress reporting
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .database import Database, FILE_MAPPING
from .datasets import LAYERS, TABLES

if TYPE_CHECKING:
    from .client import Factpages


# =============================================================================
# Retry Configuration
# =============================================================================

class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Configure retry behavior.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Multiplier for exponential backoff
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


DEFAULT_RETRY = RetryConfig()


# =============================================================================
# Sync Strategies
# =============================================================================

class SyncStrategy:
    """Base class for sync strategies."""

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        """Determine if a dataset should be synced."""
        raise NotImplementedError


class AlwaysSync(SyncStrategy):
    """Always sync - useful for initial setup or forced refresh."""

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        return True


class NeverSync(SyncStrategy):
    """Never sync - use only local data."""

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        return False


class IfMissing(SyncStrategy):
    """Sync only if dataset doesn't exist locally."""

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        return not db.has_dataset(dataset)


class IfStale(SyncStrategy):
    """Sync if dataset is older than max_age_days."""

    def __init__(self, max_age_days: int = 7):
        self.max_age_days = max_age_days

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        return db.is_stale(dataset, self.max_age_days)


class IfCountChanged(SyncStrategy):
    """
    Sync if remote record count differs from local.

    This is a lightweight way to detect changes without downloading
    all data - just compare counts.
    """

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        if not db.has_dataset(dataset):
            return True

        local_count = db.get_record_count(dataset)
        try:
            remote_count = api.get_count(dataset)
            return local_count != remote_count
        except Exception:
            # If we can't get remote count, don't sync
            return False


class IfStaleOrCountChanged(SyncStrategy):
    """
    Combination strategy: sync if stale OR if count changed.

    - If fresh (< max_age_days), check count
    - If stale, always sync
    - This balances freshness with minimizing unnecessary syncs
    """

    def __init__(self, max_age_days: int = 7):
        self.max_age_days = max_age_days

    def should_sync(self, dataset: str, db: Database, api: "Factpages") -> bool:
        if not db.has_dataset(dataset):
            return True

        # If stale, always sync
        if db.is_stale(dataset, self.max_age_days):
            return True

        # If fresh, only sync if count changed
        local_count = db.get_record_count(dataset)
        try:
            remote_count = api.get_count(dataset)
            return local_count != remote_count
        except Exception:
            return False


# =============================================================================
# Sync State (for resume capability)
# =============================================================================

class SyncState:
    """
    Tracks sync progress for resume capability.

    Saves state to disk so interrupted syncs can be resumed.
    """

    STATE_FILE = "_sync_state.json"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / self.STATE_FILE
        self._state: Optional[dict] = None

    def _load(self) -> dict:
        """Load state from disk."""
        if self._state is not None:
            return self._state

        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                self._state = json.load(f)
        else:
            self._state = {
                "in_progress": False,
                "started": None,
                "datasets_pending": [],
                "datasets_completed": [],
                "datasets_failed": [],
            }

        return self._state

    def _save(self) -> None:
        """Save state to disk."""
        if self._state:
            with open(self.state_path, 'w') as f:
                json.dump(self._state, f, indent=2)

    def start_sync(self, datasets: list[str]) -> None:
        """Mark sync as started with list of datasets to sync."""
        self._state = {
            "in_progress": True,
            "started": datetime.now().isoformat(),
            "datasets_pending": datasets.copy(),
            "datasets_completed": [],
            "datasets_failed": [],
        }
        self._save()

    def mark_completed(self, dataset: str) -> None:
        """Mark a dataset as successfully synced."""
        state = self._load()
        if dataset in state["datasets_pending"]:
            state["datasets_pending"].remove(dataset)
        if dataset not in state["datasets_completed"]:
            state["datasets_completed"].append(dataset)
        self._save()

    def mark_failed(self, dataset: str, error: str) -> None:
        """Mark a dataset as failed."""
        state = self._load()
        if dataset in state["datasets_pending"]:
            state["datasets_pending"].remove(dataset)
        state["datasets_failed"].append({"dataset": dataset, "error": error})
        self._save()

    def finish_sync(self) -> None:
        """Mark sync as complete and clean up state file."""
        self._state = None
        if self.state_path.exists():
            self.state_path.unlink()

    def has_pending(self) -> bool:
        """Check if there's an interrupted sync to resume."""
        state = self._load()
        return state.get("in_progress", False) and len(state.get("datasets_pending", [])) > 0

    def get_pending(self) -> list[str]:
        """Get list of datasets still pending."""
        state = self._load()
        return state.get("datasets_pending", [])

    def get_summary(self) -> dict:
        """Get sync state summary."""
        state = self._load()
        return {
            "in_progress": state.get("in_progress", False),
            "started": state.get("started"),
            "pending": len(state.get("datasets_pending", [])),
            "completed": len(state.get("datasets_completed", [])),
            "failed": len(state.get("datasets_failed", [])),
        }


# =============================================================================
# Sync Engine
# =============================================================================

class SyncEngine:
    """
    Orchestrates synchronization between API and local database.

    Features:
    - Multiple sync strategies
    - Retry with exponential backoff
    - Resume interrupted syncs
    - Parallel downloads
    - Smart change detection

    Example:
        >>> from factpages_py import Factpages
        >>> from factpages_py.database import Database
        >>> from factpages_py.sync import SyncEngine, IfStale
        >>>
        >>> api = Factpages()
        >>> db = api.db  # Uses ./factpages_data by default
        >>> engine = SyncEngine(api, db)
        >>>
        >>> # Sync entities if older than 7 days
        >>> engine.sync_category('entities', strategy=IfStale(max_age_days=7))
        >>>
        >>> # Resume interrupted sync
        >>> if engine.has_pending_sync():
        ...     engine.resume_sync()
    """

    def __init__(
        self,
        api: "Factpages",
        db: Database,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize the sync engine.

        Args:
            api: Factpages client for remote data
            db: Database for local storage
            retry_config: Configuration for retry behavior
        """
        self.api = api
        self.db = db
        self.retry = retry_config or DEFAULT_RETRY
        self.state = SyncState(db.data_dir)

    # =========================================================================
    # Retry Logic
    # =========================================================================

    def _download_with_retry(
        self,
        dataset: str,
        include_geometry: bool = True,
    ) -> tuple[bool, any, str]:
        """
        Download a dataset with retry logic.

        Returns:
            Tuple of (success, dataframe_or_none, error_message)
        """
        last_error = None

        for attempt in range(self.retry.max_retries + 1):
            try:
                df = self.api.download(
                    dataset,
                    include_geometry=include_geometry,
                )
                return True, df, ""

            except Exception as e:
                last_error = str(e)

                if attempt < self.retry.max_retries:
                    delay = self.retry.get_delay(attempt)
                    time.sleep(delay)

        return False, None, last_error or "Unknown error"

    # =========================================================================
    # Smart Change Detection
    # =========================================================================

    def check_for_changes(
        self,
        datasets: list[str],
        progress: bool = True
    ) -> dict[str, dict]:
        """
        Check which datasets have changes by comparing record counts.

        This is a lightweight pre-check before downloading.

        Args:
            datasets: List of datasets to check
            progress: Show progress messages

        Returns:
            Dict mapping dataset to {needs_sync, local_count, remote_count, reason}
        """
        results = {}

        if progress:
            print(f"Checking {len(datasets)} datasets for changes...")

        for dataset in datasets:
            if not self.db.has_dataset(dataset):
                results[dataset] = {
                    "needs_sync": True,
                    "local_count": 0,
                    "remote_count": None,
                    "reason": "not synced",
                }
                continue

            local_count = self.db.get_record_count(dataset)

            try:
                remote_count = self.api.get_count(dataset)
                needs_sync = local_count != remote_count

                results[dataset] = {
                    "needs_sync": needs_sync,
                    "local_count": local_count,
                    "remote_count": remote_count,
                    "reason": "count changed" if needs_sync else "up to date",
                }

            except Exception as e:
                results[dataset] = {
                    "needs_sync": False,
                    "local_count": local_count,
                    "remote_count": None,
                    "reason": f"check failed: {e}",
                }

        if progress:
            needs_sync = sum(1 for r in results.values() if r["needs_sync"])
            print(f"  {needs_sync}/{len(datasets)} datasets need sync")

        return results

    # =========================================================================
    # Single Dataset Sync
    # =========================================================================

    def sync_dataset(
        self,
        dataset: str,
        strategy: Optional[SyncStrategy] = None,
        force: bool = False,
        include_geometry: bool = True,
        progress: bool = True
    ) -> dict:
        """
        Sync a single dataset with retry logic.

        Args:
            dataset: Dataset name to sync
            strategy: SyncStrategy to use (default: IfMissing)
            force: Force sync regardless of strategy
            include_geometry: Include geometry column for spatial datasets
            progress: Show progress messages

        Returns:
            Dict with sync results
        """
        strategy = strategy or IfMissing()

        start_time = datetime.now()
        result = {
            "dataset": dataset,
            "synced": False,
            "record_count": 0,
            "duration_seconds": 0,
            "retries": 0,
            "reason": None,
        }

        # Check if sync is needed
        if not force and not strategy.should_sync(dataset, self.db, self.api):
            result["reason"] = "skipped (up to date)"
            return result

        # Download with retry
        success, df, error = self._download_with_retry(
            dataset,
            include_geometry=include_geometry,
        )

        if success and df is not None:
            self.db.put(dataset, df, source="api")
            result["synced"] = True
            result["record_count"] = len(df)
            result["reason"] = "synced"

            # Validate against expected count
            expected_count = self.db.get_remote_count(dataset)
            if expected_count is not None:
                result["expected_count"] = expected_count
                if len(df) != expected_count:
                    result["count_mismatch"] = True
                    result["count_diff"] = len(df) - expected_count
                    if progress:
                        print(f"  {dataset}: downloaded {len(df):,} records (expected {expected_count:,}, diff: {len(df) - expected_count:+,})")
        else:
            # Include table description in error message for context
            description = self.db.get_table_description(dataset)
            if description:
                result["reason"] = f"error: {error}"
                result["description"] = description
                if progress:
                    print(f"  {dataset}: FAILED - {error}")
                    print(f"    ({description})")
            else:
                result["reason"] = f"error: {error}"
                if progress:
                    print(f"  {dataset}: FAILED - {error}")

        result["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        return result

    # =========================================================================
    # Category Sync
    # =========================================================================

    def sync_category(
        self,
        category: str,
        strategy: Optional[SyncStrategy] = None,
        force: bool = False,
        progress: bool = True
    ) -> list[dict]:
        """
        Sync all datasets in a category.

        Args:
            category: One of 'entities', 'geometries', 'production', 'supporting'
            strategy: SyncStrategy to use
            force: Force sync regardless of strategy
            progress: Show progress messages

        Returns:
            List of sync results for each dataset
        """
        if category not in FILE_MAPPING:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Valid options: {list(FILE_MAPPING.keys())}"
            )

        datasets = FILE_MAPPING[category]
        strategy = strategy or IfMissing()

        if progress:
            print(f"\nSyncing {category} ({len(datasets)} datasets)")
            print("-" * 40)

        results = []
        for dataset in datasets:
            include_geometry = (category == "geometries")

            result = self.sync_dataset(
                dataset,
                strategy=strategy,
                force=force,
                include_geometry=include_geometry,
                progress=progress
            )
            results.append(result)

        if progress:
            synced = sum(1 for r in results if r["synced"])
            print(f"\nSynced {synced}/{len(datasets)} datasets")

        return results

    # =========================================================================
    # Full Sync with Resume
    # =========================================================================

    def sync_all(
        self,
        strategy: Optional[SyncStrategy] = None,
        force: bool = False,
        progress: bool = True,
        track_state: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Sync all datasets across all categories.

        Args:
            strategy: SyncStrategy to use
            force: Force sync regardless of strategy
            progress: Show progress messages
            track_state: Enable state tracking for resume capability
            workers: Number of parallel download threads (default: 4)

        Returns:
            Dict mapping category to list of sync results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        strategy = strategy or IfMissing()

        # Collect all datasets
        all_datasets = []
        for category, datasets in FILE_MAPPING.items():
            for dataset in datasets:
                all_datasets.append((category, dataset))

        if track_state:
            self.state.start_sync([d[1] for d in all_datasets])

        all_results = {cat: [] for cat in FILE_MAPPING}
        start_time = datetime.now()

        # Thread-safe lock for updating results and state
        import threading
        lock = threading.Lock()

        def sync_one(category: str, dataset: str) -> tuple[str, str, dict]:
            include_geometry = (category == "geometries")
            result = self.sync_dataset(
                dataset,
                strategy=strategy,
                force=force,
                include_geometry=include_geometry,
                progress=False
            )
            return category, dataset, result

        pbar = tqdm(
            total=len(all_datasets),
            desc="Syncing all",
            unit="table",
            disable=not progress
        )

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(sync_one, cat, ds): (cat, ds)
                    for cat, ds in all_datasets
                }

                for future in as_completed(futures):
                    category, dataset, result = future.result()

                    with lock:
                        all_results[category].append(result)

                        if track_state:
                            if result["synced"]:
                                self.state.mark_completed(dataset)
                            elif "error" in (result.get("reason") or ""):
                                self.state.mark_failed(dataset, result["reason"])
                            else:
                                self.state.mark_completed(dataset)

                    pbar.set_postfix_str(dataset)
                    pbar.update(1)

            if track_state:
                self.state.finish_sync()

        except KeyboardInterrupt:
            pbar.close()
            if progress:
                print("\n\nSync interrupted! Use resume_sync() to continue.")
            raise

        pbar.close()

        if progress:
            duration = (datetime.now() - start_time).total_seconds()
            total_synced = sum(
                1 for cat_results in all_results.values()
                for r in cat_results if r["synced"]
            )
            total_datasets = sum(len(r) for r in all_results.values())
            print(f"Completed: {total_synced}/{total_datasets} datasets in {duration:.1f}s")

        return all_results

    # =========================================================================
    # Resume Interrupted Sync
    # =========================================================================

    def has_pending_sync(self) -> bool:
        """Check if there's an interrupted sync to resume."""
        return self.state.has_pending()

    def get_sync_status(self) -> dict:
        """Get current sync state summary."""
        return self.state.get_summary()

    def resume_sync(
        self,
        strategy: Optional[SyncStrategy] = None,
        progress: bool = True
    ) -> list[dict]:
        """
        Resume an interrupted sync.

        Args:
            strategy: SyncStrategy to use (default: AlwaysSync for resume)
            progress: Show progress messages

        Returns:
            List of sync results for resumed datasets
        """
        if not self.has_pending_sync():
            if progress:
                print("No pending sync to resume")
            return []

        pending = self.state.get_pending()
        strategy = strategy or AlwaysSync()  # Always sync pending items

        if progress:
            print(f"\nResuming sync: {len(pending)} datasets remaining")
            print("-" * 40)

        results = []
        for dataset in pending:
            # Determine geometry based on dataset category
            include_geometry = any(
                dataset in FILE_MAPPING.get("geometries", [])
            )

            result = self.sync_dataset(
                dataset,
                strategy=strategy,
                force=True,
                include_geometry=include_geometry,
                progress=progress
            )
            results.append(result)

            if result["synced"]:
                self.state.mark_completed(dataset)
            elif "error" in (result.get("reason") or ""):
                self.state.mark_failed(dataset, result["reason"])

        # Check if complete
        if not self.state.get_pending():
            self.state.finish_sync()
            if progress:
                print("\nResume complete!")

        return results

    # =========================================================================
    # Parallel Sync
    # =========================================================================

    def sync_parallel(
        self,
        datasets: list[str],
        max_workers: int = 4,
        strategy: Optional[SyncStrategy] = None,
        progress: bool = True
    ) -> list[dict]:
        """
        Sync multiple datasets in parallel using threads.

        Args:
            datasets: List of datasets to sync
            max_workers: Maximum parallel downloads
            strategy: SyncStrategy to use
            progress: Show progress messages

        Returns:
            List of sync results
        """
        strategy = strategy or IfMissing()

        if progress:
            print(f"\nParallel sync: {len(datasets)} datasets with {max_workers} workers")
            print("-" * 40)

        results = []

        def sync_one(dataset: str) -> dict:
            include_geometry = dataset in FILE_MAPPING.get("geometries", [])
            return self.sync_dataset(
                dataset,
                strategy=strategy,
                include_geometry=include_geometry,
                progress=False  # Disable per-dataset progress in parallel
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(sync_one, ds): ds for ds in datasets}

            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if progress:
                        status = "OK" if result["synced"] else result.get("reason", "skipped")
                        print(f"  {dataset}: {status}")
                except Exception as e:
                    # Include table description for context
                    description = self.db.get_table_description(dataset)
                    error_result = {
                        "dataset": dataset,
                        "synced": False,
                        "reason": f"error: {e}"
                    }
                    if description:
                        error_result["description"] = description
                    results.append(error_result)
                    if progress:
                        print(f"  {dataset}: ERROR - {e}")
                        if description:
                            print(f"    ({description})")

        if progress:
            synced = sum(1 for r in results if r.get("synced"))
            print(f"\nCompleted: {synced}/{len(datasets)} synced")

        return results

    # =========================================================================
    # Smart Sync (check counts first, then download changed)
    # =========================================================================

    def smart_sync(
        self,
        datasets: Optional[list[str]] = None,
        category: Optional[str] = None,
        progress: bool = True
    ) -> list[dict]:
        """
        Smart sync: check counts first, only download changed datasets.

        This minimizes bandwidth by checking record counts before downloading.

        Args:
            datasets: Specific datasets to check (default: all)
            category: Limit to specific category
            progress: Show progress messages

        Returns:
            List of sync results for datasets that were synced
        """
        # Determine which datasets to check
        if datasets:
            to_check = datasets
        elif category:
            to_check = FILE_MAPPING.get(category, [])
        else:
            to_check = [ds for cat in FILE_MAPPING.values() for ds in cat]

        if progress:
            print(f"\nSmart sync: checking {len(to_check)} datasets")
            print("-" * 40)

        # Check which need sync
        changes = self.check_for_changes(to_check, progress=progress)
        needs_sync = [ds for ds, info in changes.items() if info["needs_sync"]]

        if not needs_sync:
            if progress:
                print("All datasets up to date!")
            return []

        if progress:
            print(f"\nDownloading {len(needs_sync)} changed datasets...")

        # Sync only changed datasets
        results = []
        for dataset in needs_sync:
            result = self.sync_dataset(
                dataset,
                strategy=AlwaysSync(),
                progress=progress
            )
            results.append(result)

        return results

    # =========================================================================
    # Data Quality & Maintenance
    # =========================================================================

    # =========================================================================
    # Stats Caching
    # =========================================================================

    STATS_CACHE_FILE = "_stats_cache.json"
    STATS_CACHE_MAX_AGE_DAYS = 3

    def _get_cached_stats(self) -> Optional[dict]:
        """Get cached stats if available and fresh."""
        cache_path = self.db.data_dir / self.STATS_CACHE_FILE
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            # Check age
            fetched_at = datetime.fromisoformat(cached.get('fetched_at', '2000-01-01'))
            age_days = (datetime.now() - fetched_at).days

            if age_days >= self.STATS_CACHE_MAX_AGE_DAYS:
                return None

            cached['cache_age_days'] = age_days
            return cached
        except Exception:
            return None

    def _save_stats_cache(self, stats: dict) -> None:
        """Save stats to cache."""
        cache_path = self.db.data_dir / self.STATS_CACHE_FILE
        stats['fetched_at'] = datetime.now().isoformat()
        with open(cache_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def _update_cached_stats_local_counts(self, cached: dict) -> dict:
        """Update cached stats with current local counts (doesn't hit API)."""
        for item in cached.get('all', []):
            dataset = item['dataset']
            if self.db.has_dataset(dataset):
                item['local_count'] = self.db.get_record_count(dataset)
                last_sync = self.db.get_last_sync(dataset)
                if last_sync:
                    item['age_days'] = (datetime.now() - last_sync).days
                    item['last_sync'] = last_sync.isoformat()
            else:
                item['local_count'] = 0
                item['age_days'] = None
                item['last_sync'] = None

            # Re-classify with updated local data
            item['status'] = self._classify_dataset(
                item['local_count'],
                item.get('remote_count'),
                item.get('age_days')
            )

        # Rebuild summary lists
        cached['fresh'] = [r for r in cached['all'] if r['status'] == 'fresh']
        cached['stale'] = [r for r in cached['all'] if r['status'] == 'stale']
        cached['changed'] = [r for r in cached['all'] if r['status'] == 'changed']
        cached['missing'] = [r for r in cached['all'] if r['status'] == 'missing']
        cached['errors'] = [r for r in cached['all'] if r['status'] == 'error']

        cached['fresh_count'] = len(cached['fresh'])
        cached['stale_count'] = len(cached['stale'])
        cached['changed_count'] = len(cached['changed'])
        cached['missing_count'] = len(cached['missing'])
        cached['error_count'] = len(cached['errors'])
        cached['total_local_records'] = sum(r['local_count'] for r in cached['all'])

        return cached

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

        Args:
            progress: Show progress messages
            workers: Number of parallel API requests
            force_refresh: Force refetch from API even if cache is fresh

        Returns:
            Dict with stats for all datasets

        Example:
            >>> stats = fp.stats()
            >>> print(f"Total remote records: {stats['total_remote_records']:,}")
            >>> print(f"Datasets with changes: {len(stats['changed'])}")
        """
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_stats()
            if cached:
                # Update local counts without hitting API
                cached = self._update_cached_stats_local_counts(cached)
                if progress:
                    print(f"Using cached stats ({cached.get('cache_age_days', 0)} days old)")
                    self._print_stats_report(cached)
                return cached

        # Fetch fresh stats from API - use actual available datasets from LAYERS and TABLES
        all_datasets = list({**LAYERS, **TABLES}.keys())

        if progress:
            print(f"Fetching stats for {len(all_datasets)} datasets from API...")

        results = []

        def get_counts(dataset: str) -> dict:
            local_count = 0
            last_sync = None
            age_days = None

            if self.db.has_dataset(dataset):
                local_count = self.db.get_record_count(dataset)
                last_sync = self.db.get_last_sync(dataset)
                if last_sync:
                    age_days = (datetime.now() - last_sync).days

            try:
                remote_count = self.api.get_count(dataset)
            except Exception:
                remote_count = None

            return {
                'dataset': dataset,
                'local_count': local_count,
                'remote_count': remote_count,
                'last_sync': last_sync.isoformat() if last_sync else None,
                'age_days': age_days,
                'status': self._classify_dataset(local_count, remote_count, age_days),
            }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(get_counts, ds): ds for ds in all_datasets}
            for future in as_completed(futures):
                results.append(future.result())

        # Classify results
        fresh = [r for r in results if r['status'] == 'fresh']
        stale = [r for r in results if r['status'] == 'stale']
        changed = [r for r in results if r['status'] == 'changed']
        missing = [r for r in results if r['status'] == 'missing']
        errors = [r for r in results if r['status'] == 'error']

        total_local = sum(r['local_count'] for r in results)
        total_remote = sum(r['remote_count'] or 0 for r in results)

        report = {
            'total_datasets': len(all_datasets),
            'total_local_records': total_local,
            'total_remote_records': total_remote,
            'fresh': fresh,
            'fresh_count': len(fresh),
            'stale': stale,
            'stale_count': len(stale),
            'changed': changed,
            'changed_count': len(changed),
            'missing': missing,
            'missing_count': len(missing),
            'errors': errors,
            'error_count': len(errors),
            'all': sorted(results, key=lambda x: x['dataset']),
        }

        # Cache the results
        self._save_stats_cache(report)

        if progress:
            self._print_stats_report(report)

        return report

    def _print_stats_report(self, report: dict) -> None:
        """Print a stats report."""
        print(f"\nDataset Statistics")
        print("-" * 50)
        print(f"Total datasets:    {report['total_datasets']}")
        print(f"Local records:     {report['total_local_records']:,}")
        print(f"Remote records:    {report['total_remote_records']:,}")
        print()
        print(f"Fresh (<30d):      {report['fresh_count']}")
        print(f"Stale (>30d):      {report['stale_count']}")
        print(f"Count changed:     {report['changed_count']}")
        print(f"Missing:           {report['missing_count']}")
        if report['error_count']:
            print(f"API errors:        {report['error_count']}")

        if report['changed']:
            print(f"\nDatasets with count changes:")
            for r in report['changed'][:5]:
                print(f"  {r['dataset']}: {r['local_count']} -> {r['remote_count']}")
            if len(report['changed']) > 5:
                print(f"  ... and {len(report['changed']) - 5} more")

    def _classify_dataset(self, local_count: int, remote_count: int, age_days: int) -> str:
        """Classify a dataset's sync status."""
        if remote_count is None:
            return 'error'
        if local_count == 0:
            return 'missing'
        if local_count != remote_count:
            return 'changed'
        if age_days is None or age_days >= 30:
            return 'stale'
        return 'fresh'

    def check_quality(self, progress: bool = True) -> dict:
        """
        Check data quality and freshness across all datasets.

        Returns a comprehensive report on:
        - Dataset freshness (age in days)
        - Missing datasets
        - Stale datasets (> 30 days old)
        - Overall health score

        Args:
            progress: Show progress messages

        Returns:
            Dict with quality report

        Example:
            >>> engine = SyncEngine(api, db)
            >>> report = engine.check_quality()
            >>> print(f"Health: {report['health_score']}%")
            >>> for ds in report['stale'][:5]:
            ...     print(f"  {ds['name']}: {ds['age_days']} days old")
        """
        all_datasets = [ds for cat in FILE_MAPPING.values() for ds in cat]
        total = len(all_datasets)

        if progress:
            print(f"Checking quality of {total} datasets...")

        fresh = []      # < 7 days
        aging = []      # 7-30 days
        stale = []      # > 30 days
        missing = []    # Not downloaded

        for dataset in all_datasets:
            if not self.db.has_dataset(dataset):
                missing.append(dataset)
                continue

            last_sync = self.db.get_last_sync(dataset)
            if not last_sync:
                missing.append(dataset)
                continue

            age = datetime.now() - last_sync
            age_days = age.days

            info = {
                'name': dataset,
                'last_sync': last_sync.isoformat(),
                'age_days': age_days,
                'record_count': self.db.get_record_count(dataset),
            }

            if age_days < 7:
                fresh.append(info)
            elif age_days < 30:
                aging.append(info)
            else:
                stale.append(info)

        # Sort stale by age (oldest first)
        stale.sort(key=lambda x: x['age_days'], reverse=True)

        # Calculate health score (0-100)
        # Fresh = 100%, Aging = 50%, Stale = 25%, Missing = 0%
        downloaded = len(fresh) + len(aging) + len(stale)
        if total > 0:
            score = (
                len(fresh) * 100 +
                len(aging) * 50 +
                len(stale) * 25
            ) / total
        else:
            score = 0

        report = {
            'total_datasets': total,
            'downloaded': downloaded,
            'missing': missing,
            'missing_count': len(missing),
            'fresh': fresh,
            'fresh_count': len(fresh),
            'aging': aging,
            'aging_count': len(aging),
            'stale': stale,
            'stale_count': len(stale),
            'health_score': round(score, 1),
            'needs_refresh': len(stale) + len(missing),
        }

        if progress:
            print(f"\nData Quality Report")
            print("-" * 40)
            print(f"Health Score:  {report['health_score']}%")
            print(f"Fresh (<7d):   {len(fresh)}")
            print(f"Aging (7-30d): {len(aging)}")
            print(f"Stale (>30d):  {len(stale)}")
            print(f"Missing:       {len(missing)}")

            if stale:
                print(f"\nOldest datasets:")
                for ds in stale[:5]:
                    print(f"  {ds['name']}: {ds['age_days']} days old")

        return report

    def refresh(
        self,
        max_age_days: int = 30,
        limit_percent: float = 10.0,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Refresh stale datasets with a limit on how many to download.

        Uses cached stats (3-day cache) to minimize API calls.

        Prioritization:
        1. Datasets with changed record counts (eagerly fetch - likely have real changes)
        2. Missing datasets (not downloaded yet)
        3. Datasets older than max_age_days (enforce freshness)

        The 10% limit ensures we don't overwhelm the API or take too long.

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
            >>> # More aggressive: refresh up to 25% of datasets
            >>> results = fp.refresh(limit_percent=25)
        """
        # Use cached stats to avoid extra API calls
        stats = self.stats(progress=False, workers=workers)

        total = stats['total_datasets']
        max_to_sync = max(1, int(total * limit_percent / 100))

        # Build priority queue from cached stats
        # Priority 1: count changed, Priority 2: missing, Priority 3: stale by age
        count_changed = []
        missing = []
        stale = []

        for item in stats['all']:
            ds = item['dataset']
            status = item['status']
            age_days = item.get('age_days') or 9999
            local = item['local_count']
            remote = item.get('remote_count')

            if status == 'changed':
                count_changed.append((ds, age_days, local, remote))
            elif status == 'missing':
                missing.append((ds, age_days))
            elif status == 'stale' or (age_days is not None and age_days >= max_age_days):
                stale.append((ds, age_days))

        # Sort each category by age (oldest first)
        count_changed.sort(key=lambda x: x[1], reverse=True)
        stale.sort(key=lambda x: x[1], reverse=True)

        # Build priority queue: count_changed first, then missing, then stale
        priority_queue = []
        for ds, age, local, remote in count_changed:
            priority_queue.append((ds, f'count: {local}->{remote}', 1))  # Priority 1
        for ds, age in missing:
            priority_queue.append((ds, 'missing', 2))  # Priority 2
        for ds, age in stale:
            priority_queue.append((ds, f'{age}d old', 3))  # Priority 3

        # Apply limit
        to_sync = [(ds, reason) for ds, reason, _ in priority_queue[:max_to_sync]]
        total_needing_sync = len(priority_queue)

        if progress:
            print(f"\nRefresh Summary:")
            print(f"  Count changed: {len(count_changed)} (priority)")
            print(f"  Missing:       {len(missing)}")
            print(f"  Stale (>{max_age_days}d): {len(stale)}")
            print(f"  Total to sync: {total_needing_sync}, syncing {len(to_sync)} (limit: {limit_percent}%)")
            if total_needing_sync > len(to_sync):
                print(f"  Skipping {total_needing_sync - len(to_sync)} datasets (will sync in future refreshes)")

        if not to_sync:
            if progress:
                print("All datasets are fresh!")
            return {
                'synced': [],
                'failed': [],
                'count_changed': 0,
                'stale_remaining': 0,
                'total_needing_sync': 0,
            }

        # Sync the selected datasets
        datasets_to_sync = [ds for ds, _ in to_sync]
        results = self.sync_parallel(datasets_to_sync, max_workers=workers, strategy=AlwaysSync(), progress=progress)

        synced = [r for r in results if r.get('synced')]
        failed = [r for r in results if not r.get('synced')]

        return {
            'synced': [r['dataset'] for r in synced],
            'synced_count': len(synced),
            'failed': [r['dataset'] for r in failed],
            'failed_count': len(failed),
            'count_changed_found': len(count_changed),
            'stale_remaining': total_needing_sync - len(to_sync),
            'total_needing_sync': total_needing_sync,
            'limit_applied': total_needing_sync > len(to_sync),
        }

    def fix(
        self,
        max_age_days: int = 30,
        include_missing: bool = True,
        progress: bool = True,
        workers: int = 4
    ) -> dict:
        """
        Thorough fix: refresh ALL stale, changed, and missing datasets without limits.

        Uses cached stats (3-day cache) to minimize API calls.

        Use this when you need a complete data refresh, like after a long
        period of inactivity or when data quality is critical.

        Syncs:
        - Datasets with changed record counts (data has changed)
        - Datasets older than max_age_days (enforce freshness)
        - Missing datasets (if include_missing=True)

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
            >>>
            >>> # Fix only stale data (don't download new datasets)
            >>> results = fp.fix(include_missing=False)
        """
        # Use cached stats to avoid extra API calls
        stats = self.stats(progress=False, workers=workers)

        count_changed = []
        stale = []
        missing = []

        for item in stats['all']:
            ds = item['dataset']
            status = item['status']
            age_days = item.get('age_days') or 9999
            local = item['local_count']
            remote = item.get('remote_count')

            if status == 'changed':
                count_changed.append((ds, local, remote))
            elif status == 'missing':
                if include_missing:
                    missing.append(ds)
            elif status == 'stale' or (age_days is not None and age_days >= max_age_days):
                stale.append((ds, age_days))

        # Combine all (no limit for fix)
        to_sync = []
        to_sync.extend((ds, f'count: {local}->{remote}') for ds, local, remote in count_changed)
        to_sync.extend((ds, 'missing') for ds in missing)
        to_sync.extend((ds, f'{age}d old') for ds, age in stale)

        if progress:
            print(f"\nFix Summary:")
            print(f"  Count changed: {len(count_changed)} (data updated)")
            print(f"  Missing:       {len(missing)}")
            print(f"  Stale (>{max_age_days}d): {len(stale)}")
            print(f"  Total to sync: {len(to_sync)}")
            print("(No limit applied - this may take a while)")

        if not to_sync:
            if progress:
                print("All datasets are fresh and complete!")
            return {
                'synced': [],
                'failed': [],
                'count_changed_found': 0,
                'total_fixed': 0,
            }

        # Sync all datasets
        datasets_to_sync = [ds for ds, _ in to_sync]
        results = self.sync_parallel(datasets_to_sync, max_workers=workers, strategy=AlwaysSync(), progress=progress)

        synced = [r for r in results if r.get('synced')]
        failed = [r for r in results if not r.get('synced')]

        # Include descriptions in failed results
        failed_info = []
        for r in failed:
            info = {'dataset': r['dataset'], 'reason': r.get('reason')}
            if r.get('description'):
                info['description'] = r['description']
            failed_info.append(info)

        return {
            'synced': [r['dataset'] for r in synced],
            'synced_count': len(synced),
            'failed': failed_info,
            'failed_count': len(failed),
            'count_changed_found': len(count_changed),
            'total_fixed': len(synced),
        }

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
            Dict with fetch results

        Example:
            >>> # Download entire database
            >>> results = fp.fetch_all()
            >>> print(f"Downloaded {results['synced_count']} datasets")
        """
        # Force refresh stats to get latest from API
        stats = self.stats(progress=progress, workers=workers, force_refresh=True)

        # Find all missing datasets
        missing = [item['dataset'] for item in stats['all'] if item['status'] == 'missing']

        if progress:
            print(f"\nFetch All Summary:")
            print(f"  Total datasets:  {stats['total_datasets']}")
            print(f"  Already have:    {stats['total_datasets'] - len(missing)}")
            print(f"  Missing:         {len(missing)}")

        if not missing:
            if progress:
                print("\nDatabase is complete - all datasets already downloaded!")
            return {
                'synced': [],
                'synced_count': 0,
                'failed': [],
                'failed_count': 0,
                'already_had': stats['total_datasets'],
                'total_datasets': stats['total_datasets'],
            }

        if progress:
            print(f"\nDownloading {len(missing)} missing datasets...")

        # Download all missing datasets
        results = self.sync_parallel(missing, max_workers=workers, strategy=AlwaysSync(), progress=progress)

        synced = [r for r in results if r.get('synced')]
        failed = [r for r in results if not r.get('synced')]

        # Include descriptions in failed results
        failed_info = []
        for r in failed:
            info = {'dataset': r['dataset'], 'reason': r.get('reason')}
            if r.get('description'):
                info['description'] = r['description']
            failed_info.append(info)

        return {
            'synced': [r['dataset'] for r in synced],
            'synced_count': len(synced),
            'failed': failed_info,
            'failed_count': len(failed),
            'already_had': stats['total_datasets'] - len(missing),
            'total_datasets': stats['total_datasets'],
        }

    # =========================================================================
    # Legacy Methods (backwards compatibility)
    # =========================================================================

    def check_updates(self, progress: bool = True) -> dict:
        """Check which datasets have updates available."""
        all_datasets = [ds for cat in FILE_MAPPING.values() for ds in cat]
        return self.check_for_changes(all_datasets, progress=progress)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_sync(
    data_dir: str = "./factpages_data",
    categories: Optional[list[str]] = None,
    max_age_days: int = 7,
    progress: bool = True
) -> Database:
    """
    Quick sync function for common use cases.

    Downloads missing or stale data and returns a ready-to-use database.

    Args:
        data_dir: Directory for local database
        categories: Categories to sync (default: all)
        max_age_days: Max age before data is considered stale
        progress: Show progress

    Returns:
        Database instance with synced data

    Example:
        >>> from factpages_py.sync import quick_sync
        >>> db = quick_sync(categories=['entities'])
        >>> discoveries = db.get('discovery')
    """
    from .client import Factpages

    api = Factpages(data_dir=data_dir)
    db = api.db
    engine = SyncEngine(api, db)

    strategy = IfStale(max_age_days=max_age_days)
    categories = categories or list(FILE_MAPPING.keys())

    for category in categories:
        if category in FILE_MAPPING:
            engine.sync_category(category, strategy=strategy, progress=progress)

    return db
