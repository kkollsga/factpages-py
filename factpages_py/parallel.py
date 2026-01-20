"""
Parallel Fetching Utilities

Single source of truth for all parallel/concurrent data operations.
Provides consistent patterns for:
- Multi-threaded API requests
- Dataset synchronization
- Progress reporting (tqdm)
- Error handling
- Thread-safe state management

Usage:
    from .parallel import parallel_fetch, parallel_sync

    # Simple parallel fetch
    results = parallel_fetch(
        tasks={'field': lambda: fetch('field'), 'discovery': lambda: fetch('discovery')},
        workers=4,
        progress=True,
        desc="Downloading"
    )

    # Parallel dataset sync with metadata returns
    results = parallel_sync(
        datasets=['field', 'discovery'],
        sync_fn=lambda ds: engine.sync_dataset(ds),
        workers=4,
        progress=True
    )
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

T = TypeVar('T')


class ParallelResult:
    """
    Result container for parallel operations.

    Provides both dict-like and attribute access to results,
    with built-in success/failure tracking.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add_success(self, key: str, result: Any, elapsed: float = 0.0) -> None:
        """Thread-safe add of successful result."""
        with self._lock:
            self.results[key] = result
            self.timings[key] = elapsed

    def add_error(self, key: str, error: str, elapsed: float = 0.0) -> None:
        """Thread-safe add of error result."""
        with self._lock:
            self.errors[key] = error
            self.timings[key] = elapsed

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return self.success_count + self.error_count

    def __getitem__(self, key: str) -> Any:
        return self.results.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.results

    def __iter__(self):
        return iter(self.results)

    def items(self):
        return self.results.items()

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()


def parallel_fetch(
    tasks: Dict[str, Callable[[], T]],
    workers: int = 4,
    progress: bool = False,
    desc: str = "Processing",
    unit: str = "item"
) -> ParallelResult:
    """
    Execute multiple tasks in parallel with consistent handling.

    This is the primary function for parallel operations - all other
    parallel code should use this as the foundation.

    Args:
        tasks: Dict mapping task names to callables (no args)
        workers: Max parallel workers (default: 4)
        progress: Show tqdm progress bar
        desc: Progress bar description
        unit: Progress bar unit (e.g., "table", "item")

    Returns:
        ParallelResult with results, errors, and timings

    Example:
        >>> results = parallel_fetch({
        ...     'field': lambda: client.download('field'),
        ...     'discovery': lambda: client.download('discovery'),
        ... }, workers=4, progress=True, desc="Downloading")
        >>> print(f"Success: {results.success_count}, Errors: {results.error_count}")
    """
    result = ParallelResult()

    if not tasks:
        return result

    def execute_task(name: str, task: Callable) -> Tuple[str, Any, float, Optional[str]]:
        """Execute single task with timing and error capture."""
        start = time.time()
        try:
            task_result = task()
            elapsed = time.time() - start
            return name, task_result, elapsed, None
        except Exception as e:
            elapsed = time.time() - start
            return name, None, elapsed, str(e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute_task, name, task)
            for name, task in tasks.items()
        ]

        # Create iterator (with or without tqdm)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=desc,
                    unit=unit
                )
            except ImportError:
                iterator = as_completed(futures)
        else:
            iterator = as_completed(futures)

        # Collect results
        for future in iterator:
            name, task_result, elapsed, error = future.result()
            if error:
                result.add_error(name, error, elapsed)
            else:
                result.add_success(name, task_result, elapsed)

    return result


def parallel_sync(
    datasets: List[str],
    sync_fn: Callable[[str], dict],
    workers: int = 4,
    progress: bool = False,
    desc: str = "Syncing",
    get_expected_fn: Optional[Callable[[str], int]] = None
) -> Dict[str, dict]:
    """
    Sync multiple datasets in parallel with detailed metadata.

    Specialized for dataset synchronization with metadata tracking.

    Args:
        datasets: List of dataset names to sync
        sync_fn: Function that syncs a single dataset, returns dict with 'synced', 'record_count'
        workers: Max parallel workers
        progress: Show tqdm progress bar
        desc: Progress bar description
        get_expected_fn: Optional function to get expected record count for a dataset

    Returns:
        Dict mapping dataset names to result metadata:
        {
            'dataset_name': {
                'dataset': 'dataset_name',
                'status': 'synced' | 'failed',
                'actual_records': int,
                'expected_records': int,
                'records_match': bool,
                'duration_seconds': float,
                'error': str | None
            }
        }

    Example:
        >>> results = parallel_sync(
        ...     datasets=['field', 'discovery'],
        ...     sync_fn=lambda ds: engine.sync_dataset(ds),
        ...     workers=4,
        ...     progress=True,
        ...     get_expected_fn=lambda ds: db.get_remote_count(ds)
        ... )
    """
    if not datasets:
        return {}

    def sync_one(dataset: str) -> Tuple[str, dict, float]:
        """Sync single dataset with timing."""
        start = time.time()
        try:
            result = sync_fn(dataset)
            elapsed = time.time() - start
            return dataset, result, elapsed
        except Exception as e:
            elapsed = time.time() - start
            return dataset, {'synced': False, 'error': str(e)}, elapsed

    results = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sync_one, ds) for ds in datasets]

        # Create iterator
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=desc,
                    unit="table"
                )
            except ImportError:
                iterator = as_completed(futures)
        else:
            iterator = as_completed(futures)

        # Collect results with metadata
        for future in iterator:
            dataset, result, elapsed = future.result()

            actual_count = result.get('record_count', 0)
            expected = get_expected_fn(dataset) if get_expected_fn else 0
            error_msg = result.get('error') if not result.get('synced') else None

            # Print error message for failed syncs
            if error_msg and progress:
                # Use tqdm.write to avoid breaking progress bar
                try:
                    from tqdm import tqdm
                    tqdm.write(f"  {dataset}: FAILED - {error_msg}")
                except ImportError:
                    print(f"  {dataset}: FAILED - {error_msg}")

            results[dataset] = {
                'dataset': dataset,
                'status': 'synced' if result.get('synced') else 'failed',
                'actual_records': actual_count,
                'expected_records': expected,
                'records_match': actual_count == expected if expected else True,
                'duration_seconds': round(elapsed, 3),
                'error': error_msg
            }

    return results


def parallel_count(
    datasets: List[str],
    count_fn: Callable[[str], int],
    workers: int = 8
) -> Dict[str, int]:
    """
    Get record counts for multiple datasets in parallel.

    Optimized for fast count fetching with higher worker count.

    Args:
        datasets: List of dataset names
        count_fn: Function that returns count for a dataset
        workers: Max parallel workers (default: 8, higher for count operations)

    Returns:
        Dict mapping dataset names to counts (excludes failed counts)

    Example:
        >>> counts = parallel_count(
        ...     datasets=['field', 'discovery', 'wellbore'],
        ...     count_fn=lambda ds: client.get_count(ds),
        ...     workers=8
        ... )
    """
    if not datasets:
        return {}

    def fetch_count(dataset: str) -> Tuple[str, int]:
        try:
            count = count_fn(dataset)
            return dataset, count
        except Exception:
            return dataset, -1  # Mark as failed

    counts = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_count, ds): ds for ds in datasets}

        for future in as_completed(futures):
            dataset, count = future.result()
            if count >= 0:
                counts[dataset] = count

    return counts


def parallel_map(
    items: List[T],
    fn: Callable[[T], Any],
    workers: int = 4,
    progress: bool = False,
    desc: str = "Processing"
) -> List[Tuple[T, Any, Optional[str]]]:
    """
    Apply a function to items in parallel (like map but parallel).

    Args:
        items: List of items to process
        fn: Function to apply to each item
        workers: Max parallel workers
        progress: Show tqdm progress bar
        desc: Progress bar description

    Returns:
        List of (item, result, error) tuples in completion order

    Example:
        >>> results = parallel_map(
        ...     items=[1, 2, 3, 4, 5],
        ...     fn=lambda x: x * 2,
        ...     workers=4
        ... )
    """
    if not items:
        return []

    def process_item(item: T) -> Tuple[T, Any, Optional[str]]:
        try:
            result = fn(item)
            return item, result, None
        except Exception as e:
            return item, None, str(e)

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_item, item) for item in items]

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=desc
                )
            except ImportError:
                iterator = as_completed(futures)
        else:
            iterator = as_completed(futures)

        for future in iterator:
            results.append(future.result())

    return results


class ParallelExecutor:
    """
    Reusable parallel executor with configuration.

    Use when you need to run multiple parallel operations
    with the same configuration.

    Example:
        >>> executor = ParallelExecutor(workers=4, progress=True)
        >>> result1 = executor.fetch(tasks1)
        >>> result2 = executor.fetch(tasks2)
    """

    def __init__(
        self,
        workers: int = 4,
        progress: bool = True,
        desc: str = "Processing"
    ):
        self.workers = workers
        self.progress = progress
        self.desc = desc

    def fetch(
        self,
        tasks: Dict[str, Callable[[], T]],
        desc: Optional[str] = None
    ) -> ParallelResult:
        """Execute tasks with this executor's configuration."""
        return parallel_fetch(
            tasks=tasks,
            workers=self.workers,
            progress=self.progress,
            desc=desc or self.desc
        )

    def sync(
        self,
        datasets: List[str],
        sync_fn: Callable[[str], dict],
        get_expected_fn: Optional[Callable[[str], int]] = None,
        desc: Optional[str] = None
    ) -> Dict[str, dict]:
        """Sync datasets with this executor's configuration."""
        return parallel_sync(
            datasets=datasets,
            sync_fn=sync_fn,
            workers=self.workers,
            progress=self.progress,
            desc=desc or self.desc,
            get_expected_fn=get_expected_fn
        )

    def count(
        self,
        datasets: List[str],
        count_fn: Callable[[str], int]
    ) -> Dict[str, int]:
        """Get counts with this executor's configuration."""
        return parallel_count(
            datasets=datasets,
            count_fn=count_fn,
            workers=self.workers
        )

    def map(
        self,
        items: List[T],
        fn: Callable[[T], Any],
        desc: Optional[str] = None
    ) -> List[Tuple[T, Any, Optional[str]]]:
        """Map function over items with this executor's configuration."""
        return parallel_map(
            items=items,
            fn=fn,
            workers=self.workers,
            progress=self.progress,
            desc=desc or self.desc
        )
