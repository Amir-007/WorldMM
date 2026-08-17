"""
Append-only checkpointing so long builds are resumable and idempotent.

The stock pipeline accumulates everything in memory and writes one large JSON
at the very end, so a crash at hour 30 of a 30-hour run costs all 30 hours.
`CheckpointStore` instead appends one JSON line per completed unit of work and
flushes it to disk immediately, so a re-run skips whatever already finished.

Keys must be stable across runs. Content hashes (`chunk-<md5>`) and timestamp
keys both qualify; list positions do not.

    store = CheckpointStore("openie.progress.jsonl")
    for chunk in store.pending(chunks, key=lambda c: c.chunk_key):
        store.record(chunk.chunk_key, run_openie(chunk))
    store.write_json("openie_results.json", transform=combine)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CheckpointStore:
    """A resumable key/value log backed by a JSONL file."""

    def __init__(self, path: str, *, fsync: bool = True) -> None:
        """
        Args:
            path: JSONL progress file. Created on first `record`.
            fsync: fsync after every record. Costs a few ms per unit of work,
                which is noise next to a multi-second LLM call, and is what
                makes a hard kill survivable. Turn off only for cheap inner
                loops where losing the tail is acceptable.
        """
        self.path = path
        self.fsync = fsync
        self._records: Dict[str, Any] = {}
        self._handle = None
        self._load()

    def _load(self) -> None:
        """
        Read an existing progress file, repairing a partial trailing record.

        A hard kill can leave the final line half-written with no newline. That
        byte range is truncated off here rather than merely skipped: leaving it
        in place would make the next `record` append onto the partial line,
        corrupting that record too and losing it on every subsequent resume.
        """
        if not os.path.exists(self.path):
            return

        with open(self.path, "rb") as f:
            raw = f.read()
        if not raw:
            return

        # Everything up to the last newline is complete; anything after it is a
        # record that was still being written when the process died.
        complete_len = raw.rfind(b"\n") + 1
        partial = raw[complete_len:]

        corrupt = 0
        for lineno, line in enumerate(raw[:complete_len].splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
                self._records[entry["key"]] = entry["value"]
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                logger.error("Corrupt record at %s:%d, skipping", self.path, lineno)
                corrupt += 1

        if partial:
            logger.warning(
                "Discarding %d truncated byte(s) at the end of %s (interrupted run)",
                len(partial), self.path,
            )
            os.truncate(self.path, complete_len)

        logger.info(
            "Resuming from %s: %d completed%s",
            self.path, len(self._records),
            f", {corrupt} corrupt record(s) skipped" if corrupt else "",
        )

    def _open(self):
        if self._handle is None:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
        return self._handle

    def record(self, key: str, value: Any) -> None:
        """Mark one unit of work complete and persist it immediately."""
        handle = self._open()
        handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        self._records[key] = value

    def completed(self) -> Set[str]:
        """Keys already finished."""
        return set(self._records)

    def get(self, key: str, default: Any = None) -> Any:
        return self._records.get(key, default)

    def results(self) -> Dict[str, Any]:
        """All recorded values, in the order they were first completed."""
        return dict(self._records)

    def pending(self, items: Iterable[T], key: Callable[[T], str]) -> Iterator[T]:
        """Yield only the items whose key has not been recorded yet."""
        done = self.completed()
        skipped = 0
        for item in items:
            if key(item) in done:
                skipped += 1
                continue
            yield item
        if skipped:
            logger.info("Skipped %d already-completed item(s)", skipped)

    def compact(self) -> int:
        """
        Rewrite the log with one line per key, dropping superseded duplicates.

        Written to a temp file and moved into place, so an interrupted compact
        leaves the original intact.
        """
        self.close()
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for key, value in self._records.items():
                    f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return len(self._records)

    def write_json(
        self,
        output_path: str,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        *,
        order: Optional[List[str]] = None,
        indent: int = 2,
    ) -> None:
        """
        Materialise the finished results as a single JSON file.

        Args:
            output_path: Destination, written atomically.
            transform: Optional shaper, e.g. to split into the `ner_results` /
                `triple_results` layout the downstream stages expect.
            order: Key order for the output. Defaults to completion order;
                pass the original input order for stable, diffable artifacts.
        """
        records = self._records
        if order is not None:
            missing = [k for k in order if k not in records]
            if missing:
                logger.warning(
                    "%d key(s) in the requested order were never completed, e.g. %s",
                    len(missing), missing[:3],
                )
            records = {k: records[k] for k in order if k in records}

        payload = transform(records) if transform else records

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=indent, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, output_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        logger.info("Wrote %d record(s) to %s", len(records), output_path)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
