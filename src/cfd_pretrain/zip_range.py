"""HTTP-range reader for extracting selected members of a large ZIP archive.

The CFDBench archives are multi-gigabyte files.  This module reads the ZIP
central directory and exact local members, so the pipeline can download a
small, auditable case subset without materializing the complete archive.
"""

from __future__ import annotations

import io
import struct
import time
import warnings
import zlib
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    local_header_offset: int


class RangeZipReader:
    def __init__(self, url: str, timeout: int = 120) -> None:
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.tls_verify = True
        self._fallback_used = False
        self.total_size = self._head_size()

    def _request(self, method: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("allow_redirects", True)
        kwargs.setdefault("verify", self.tls_verify)
        try:
            response = self.session.request(method, self.url, **kwargs)
        except requests.exceptions.SSLError:
            if self._fallback_used:
                raise
            warnings.warn("TLS verification failed; retrying with verification disabled for this download.")
            self.tls_verify = False
            self._fallback_used = True
            kwargs["verify"] = False
            response = self.session.request(method, self.url, **kwargs)
        response.raise_for_status()
        return response

    def _head_size(self) -> int:
        response = self._request("HEAD")
        value = response.headers.get("Content-Length")
        if value is None:
            raise RuntimeError(f"No Content-Length returned for {self.url}")
        return int(value)

    def get_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.total_size:
            raise ValueError(f"Invalid range {start}-{end} for {self.total_size} bytes")
        expected = end - start + 1
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._request("GET", headers={"Range": f"bytes={start}-{end}"})
                data = response.content
                if response.status_code != 206:
                    raise RuntimeError(f"Server did not honor Range request: {response.status_code}")
                if len(data) != expected:
                    raise RuntimeError(f"Short range response: expected {expected}, got {len(data)}")
                return data
            except (requests.exceptions.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Range request failed after retries for bytes={start}-{end}: {last_error}") from last_error

    def get_range_chunked(self, start: int, end: int, chunk_size: int = 8 * 1024 * 1024) -> bytes:
        """Read a range in bounded chunks to tolerate proxy timeouts on large members."""

        if end < start:
            return b""
        chunks = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + chunk_size - 1)
            chunks.append(self.get_range(cursor, chunk_end))
            cursor = chunk_end + 1
        return b"".join(chunks)

    def entries(self) -> list[ZipEntry]:
        tail_size = min(self.total_size, 1024 * 1024)
        tail_start = self.total_size - tail_size
        tail = self.get_range(tail_start, self.total_size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise RuntimeError("ZIP end-of-central-directory record not found in archive tail")
        _, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_len = struct.unpack_from(
            "<4s4H2IH", tail, eocd
        )
        if disk != 0 or cd_disk != 0:
            raise RuntimeError("Multi-disk ZIP archives are not supported")
        if total_entries != disk_entries:
            raise RuntimeError("Unexpected ZIP entry count")
        central = self.get_range(cd_offset, cd_offset + cd_size - 1)
        entries: list[ZipEntry] = []
        offset = 0
        for _ in range(total_entries):
            if central[offset : offset + 4] != b"PK\x01\x02":
                raise RuntimeError(f"Invalid central-directory entry at offset {offset}")
            values = struct.unpack_from("<4s6H3I5H2I", central, offset)
            _, _, _, _, compression, _, _, crc32, comp_size, uncomp_size, name_len, extra_len, comment_len, _, _, _, local_offset = values
            start = offset + 46
            name = central[start : start + name_len].decode("utf-8", errors="replace")
            entries.append(
                ZipEntry(
                    name=name,
                    compression=compression,
                    compressed_size=comp_size,
                    uncompressed_size=uncomp_size,
                    crc32=crc32,
                    local_header_offset=local_offset,
                )
            )
            offset += 46 + name_len + extra_len + comment_len
        return entries

    def read_entry(self, entry: ZipEntry) -> bytes:
        header = self.get_range(entry.local_header_offset, entry.local_header_offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local header for {entry.name}")
        _, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<4s5H3I2H", header)
        start = entry.local_header_offset + 30 + name_len + extra_len
        compressed = self.get_range_chunked(start, start + entry.compressed_size - 1) if entry.compressed_size else b""
        if entry.compression == 0:
            payload = compressed
        elif entry.compression == 8:
            payload = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"Unsupported ZIP compression method {entry.compression} for {entry.name}")
        if len(payload) != entry.uncompressed_size:
            raise RuntimeError(f"Size mismatch for {entry.name}")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != entry.crc32:
            raise RuntimeError(f"CRC mismatch for {entry.name}")
        return payload

    def audit(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "archive_size_bytes": self.total_size,
            "range_requests": True,
            "tls_verification_used": self.tls_verify,
            "tls_fallback_used": self._fallback_used,
        }
