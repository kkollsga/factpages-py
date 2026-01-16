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
        progress: bool = True
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
                    progress=progress
                )
                return True, df, ""

            except Exception as e:
                last_error = str(e)

                if attempt < self.retry.max_retries:
                    delay = self.retry.get_delay(attempt)
                    if progress:
                        print(f"  Retry {attempt + 1}/{self.retry.max_retries} "
                              f"in {delay:.1f}s: {e}")
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
            if progress:
                print(f"  {dataset}: skipped (up to date)")
            return result

        # Download with retry
        success, df, error = self._download_with_retry(
            dataset,
            include_geometry=include_geometry,
            progress=progress
        )

        if success and df is not None:
            self.db.put(dataset, df, source="api")
            result["synced"] = True
            result["record_count"] = len(df)
            result["reason"] = "synced"
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
        track_state: bool = True
    ) -> dict:
        """
        Sync all datasets across all categories.

        Args:
            strategy: SyncStrategy to use
            force: Force sync regardless of strategy
            progress: Show progress messages
            track_state: Enable state tracking for resume capability

        Returns:
            Dict mapping category to list of sync results
        """
        strategy = strategy or IfMissing()

        # Collect all datasets
        all_datasets = []
        for category, datasets in FILE_MAPPING.items():
            for dataset in datasets:
                all_datasets.append((category, dataset))

        if track_state:
            self.state.start_sync([d[1] for d in all_datasets])

        if progress:
            print("\n" + "=" * 60)
            print("FULL DATABASE SYNC")
            print("=" * 60)

        all_results = {cat: [] for cat in FILE_MAPPING}
        start_time = datetime.now()

        try:
            for category, dataset in all_datasets:
                include_geometry = (category == "geometries")

                result = self.sync_dataset(
                    dataset,
                    strategy=strategy,
                    force=force,
                    include_geometry=include_geometry,
                    progress=progress
                )
                all_results[category].append(result)

                if track_state:
                    if result["synced"]:
                        self.state.mark_completed(dataset)
                    elif "error" in (result.get("reason") or ""):
                        self.state.mark_failed(dataset, result["reason"])
                    else:
                        self.state.mark_completed(dataset)  # Skipped counts as done

            if track_state:
                self.state.finish_sync()

        except KeyboardInterrupt:
            if progress:
                print("\n\nSync interrupted! Use resume_sync() to continue.")
            raise

        if progress:
            duration = (datetime.now() - start_time).total_seconds()
            total_synced = sum(
                1 for cat_results in all_results.values()
                for r in cat_results if r["synced"]
            )
            total_datasets = sum(len(r) for r in all_results.values())
            print(f"\n{'=' * 60}")
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
                    results.append({
                        "dataset": dataset,
                        "synced": False,
                        "reason": f"error: {e}"
                    })
                    if progress:
                        print(f"  {dataset}: ERROR - {e}")

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
