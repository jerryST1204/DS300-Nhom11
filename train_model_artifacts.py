"""Fit and serialize the current content-based recommendation model.

The available user/song history is explicitly synthetic. It is not used by
this trainer, and no collaborative-filtering model is exported.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .data_generator import DEFAULT_CATALOG, load_song_catalog
from .models import ContentBasedRecommender


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "trained_artifacts"
CONTENT_OPTIONS = {"genre_weight": 1.0, "text_weight": 1.0, "lyrics_weight": 2.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(model: object, path: Path) -> None:
    started = time.perf_counter()
    joblib.dump(model, path, compress=3)
    print(
        f"Saved {path.name}: {path.stat().st_size:,} bytes "
        f"({time.perf_counter() - started:.1f}s)"
    )


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    songs = load_song_catalog(DEFAULT_CATALOG)
    approved_lyrics = int(songs["has_lyrics"].sum())

    catalog_columns = [
        "song_id",
        "track_id",
        "title",
        "artist",
        "dataset_source",
        "genre",
        "genre_confidence",
        "genre_source",
        "has_lyrics",
        "duration",
        "view_count",
        "like_count",
        "upload_date",
    ]
    catalog = songs[[column for column in catalog_columns if column in songs.columns]]
    catalog_path = ARTIFACTS_DIR / "catalog.csv"
    catalog.to_csv(catalog_path, index=False, encoding="utf-8")

    track_ids = songs["track_id"].astype(str).tolist()
    mappings = {
        "track_id_to_song_id": {
            track_id: int(song_id)
            for track_id, song_id in zip(track_ids, songs["song_id"])
        },
        "song_id_to_track_id": track_ids,
    }
    mappings_path = ARTIFACTS_DIR / "mappings.json"
    mappings_path.write_text(
        json.dumps(mappings, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(
        f"Fitting content model on {len(songs):,} tracks; "
        f"{approved_lyrics:,} have approved lyrics."
    )
    content_model = ContentBasedRecommender(**CONTENT_OPTIONS).fit(songs)
    # The fitted sparse vectors retain the lyric information needed for search.
    # Drop source text from the serialized catalog to keep runtime artifacts small.
    content_model.songs = content_model.songs.drop(
        columns=[
            column
            for column in (
                "lyrics",
                "lyrics_clean",
                "subtitle_clean",
                "lyrics_source",
                "lyrics_status",
                "lyrics_quality",
                "match_score",
                "lyrics_use_for_vector",
                "genre_confidence",
                "genre_source",
            )
            if column in content_model.songs.columns
        ]
    )
    model_path = ARTIFACTS_DIR / "content_based.joblib"
    _dump(content_model, model_path)

    artifact_names = ["catalog.csv", "mappings.json", "content_based.joblib"]
    manifest = {
        "format_version": 2,
        "dataset": {
            "catalog_file": str(DEFAULT_CATALOG.relative_to(PROJECT_DIR)),
            "catalog_sha256": _sha256(DEFAULT_CATALOG),
            "n_tracks": int(len(songs)),
            "n_tracks_with_approved_lyrics": approved_lyrics,
            "sources": {
                str(source): int(count)
                for source, count in (
                    songs.get(
                        "dataset_source", pd.Series("unknown", index=songs.index)
                    )
                    .fillna("unknown")
                    .value_counts()
                    .items()
                )
            },
            "interaction_history_used": False,
        },
        "training": {
            "content_based": CONTENT_OPTIONS,
            "lyrics_policy": "lyrics_use_for_vector must be true",
            "genre_policy": {
                "confidence_weights": {
                    "high": 1.0,
                    "medium": 0.75,
                    "low": 0.25,
                    "very_low": 0.1,
                    "unknown": 0.0,
                },
                "description": "existing genre label weighted by genre confidence",
            },
            "subtitle_policy": "not used as a substitute for song lyrics",
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sha256": {
            name: _sha256(ARTIFACTS_DIR / name) for name in artifact_names
        },
    }
    (ARTIFACTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Content-based artifacts ready.")
    print(f"Tracks: {len(songs):,}; approved lyrics: {approved_lyrics:,}")
    print("Synthetic history and collaborative filtering were not used.")


if __name__ == "__main__":
    main()
