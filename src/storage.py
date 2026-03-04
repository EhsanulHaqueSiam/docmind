from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import settings


class StorageBackend(Protocol):
    def list_files(self) -> list[Path]: ...
    def read_file(self, path: str | Path) -> bytes: ...
    def save_file(self, filename: str, data: bytes) -> Path: ...
    def delete_file(self, path: str | Path) -> None: ...


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".txt",
    ".md",
    ".csv",
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


class LocalStorage:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> list[Path]:
        return [
            f
            for f in self.base_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def read_file(self, path: str | Path) -> bytes:
        return Path(path).read_bytes()

    def save_file(self, filename: str, data: bytes) -> Path:
        dest = self.base_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def delete_file(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)


class GCSStorage:
    def __init__(self, bucket_name: str) -> None:
        from google.cloud import storage

        client = storage.Client(project=settings.gcp_project_id)
        self.bucket = client.bucket(bucket_name)
        self._local_cache = Path("/tmp/rag_gcs_cache")
        self._local_cache.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> list[Path]:
        paths = []
        for blob in self.bucket.list_blobs():
            suffix = Path(blob.name).suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS:
                paths.append(Path(blob.name))
        return paths

    def read_file(self, path: str | Path) -> bytes:
        blob = self.bucket.blob(str(path))
        return blob.download_as_bytes()

    def save_file(self, filename: str, data: bytes) -> Path:
        blob = self.bucket.blob(filename)
        blob.upload_from_string(data)
        # Cache locally for processing
        local = self._local_cache / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return local

    def delete_file(self, path: str | Path) -> None:
        blob = self.bucket.blob(str(path))
        blob.delete()
        (self._local_cache / str(path)).unlink(missing_ok=True)


_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        if settings.storage_mode == "gcs":
            _storage = GCSStorage(settings.gcs_bucket)
        else:
            _storage = LocalStorage(settings.docs_directory)
    return _storage
