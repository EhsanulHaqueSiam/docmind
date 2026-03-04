from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config import settings
from src.storage import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class _IngestHandler(FileSystemEventHandler):
    """Debounced file event handler that triggers ingestion."""

    def __init__(self, debounce_seconds: float | None = None) -> None:
        super().__init__()
        self._debounce = debounce_seconds or settings.watcher_debounce_seconds
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _schedule_processing(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._process_pending)
            self._timer.daemon = True
            self._timer.start()

    def _process_pending(self) -> None:
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()

        from src.ingest import ingest_file

        for path_str in paths:
            path = Path(path_str)
            if path.exists() and path.is_file():
                try:
                    result = ingest_file(path)
                    logger.info("Auto-ingested %s: %s", path.name, result["status"])
                except Exception as e:
                    logger.error("Auto-ingest failed for %s: %s", path.name, e)

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            self._pending[str(path)] = time.time()
        self._schedule_processing()


class DocWatcher:
    """Watches a directory for new/modified documents and auto-ingests them."""

    def __init__(self, watch_dir: str | Path | None = None) -> None:
        self.watch_dir = Path(watch_dir or settings.docs_directory)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._observer: Observer | None = None

    def start(self) -> None:
        handler = _IngestHandler()
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info("File watcher started on: %s", self.watch_dir)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            logger.info("File watcher stopped")
