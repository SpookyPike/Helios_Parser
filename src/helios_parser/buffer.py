from __future__ import annotations

import codecs
import mmap
from dataclasses import dataclass
from pathlib import Path

from .model import TextSpan


@dataclass(slots=True)
class HeliosBuffer:
    source: Path
    access_mode: str
    _raw: bytes | mmap.mmap
    encoding: str = "utf-8"
    _file_handle: object | None = None
    _mmap_handle: mmap.mmap | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        access_mode: str = "memory",
        encoding: str = "utf-8",
    ) -> "HeliosBuffer":
        source = Path(path)
        if access_mode == "memory":
            data = source.read_bytes()
            return cls(source=source, access_mode=access_mode, _raw=data, encoding=encoding)
        if access_mode == "mmap":
            file_handle = source.open("rb")
            file_size = source.stat().st_size
            if file_size == 0:
                file_handle.close()
                return cls(source=source, access_mode=access_mode, _raw=b"", encoding=encoding)
            mmap_handle = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            return cls(
                source=source,
                access_mode=access_mode,
                _raw=mmap_handle,
                encoding=encoding,
                _file_handle=file_handle,
                _mmap_handle=mmap_handle,
            )
        raise ValueError(f"Unsupported access_mode={access_mode!r}")

    @property
    def raw(self) -> bytes | mmap.mmap:
        return self._raw

    def __len__(self) -> int:
        return len(self._raw)

    def slice_bytes(self, span: TextSpan) -> memoryview:
        return memoryview(self._raw)[span.start : span.stop]

    def slice_text(self, span: TextSpan) -> str:
        return codecs.decode(self.slice_bytes(span), self.encoding, "replace")

    def tail_text(self, max_bytes: int) -> str:
        start = max(0, len(self) - max(0, int(max_bytes)))
        return codecs.decode(memoryview(self._raw)[start:], self.encoding, "replace")

    def slice(self, span: TextSpan) -> str:
        return self.slice_text(span)

    def close(self) -> None:
        if self._mmap_handle is not None:
            self._mmap_handle.close()
            self._mmap_handle = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> "HeliosBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
