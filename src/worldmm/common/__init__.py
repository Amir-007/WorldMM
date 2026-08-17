"""Shared building blocks for the Track A pipeline: timestamps, checkpointing, mapping.

Deliberately free of torch/igraph imports so it runs on a laptop as well as on
the cluster.
"""

from .checkpoint import CheckpointStore
from .jsonrepair import fix_broken_generated_json, salvage_json
from .schema_coercion import coerce_to_schema
from .mapping import (
    DEFAULT_EXCLUDED_DAYS,
    CaptionChunk,
    VerificationReport,
    build_caption_chunks,
    chunk_key,
    load_caption_chunks,
    timestamp_to_chunk_key,
    triples_by_timestamp,
    verify,
)
from .timestamps import (
    EgoTimestamp,
    chunk_timestamp_key,
    day_of,
    duration_seconds,
    format_key,
    parse,
    to_seconds,
)

__all__ = [
    "CheckpointStore",
    "CaptionChunk",
    "DEFAULT_EXCLUDED_DAYS",
    "EgoTimestamp",
    "VerificationReport",
    "build_caption_chunks",
    "chunk_key",
    "chunk_timestamp_key",
    "coerce_to_schema",
    "day_of",
    "duration_seconds",
    "fix_broken_generated_json",
    "format_key",
    "salvage_json",
    "load_caption_chunks",
    "parse",
    "timestamp_to_chunk_key",
    "to_seconds",
    "triples_by_timestamp",
    "verify",
]
