"""Small application-facing wrapper around the content-based model."""

from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__:
    from .data_generator import (
        DEFAULT_CATALOG,
        load_song_catalog,
        load_user_history,
    )
    from .models import ContentBasedRecommender
else:
    from data_generator import (
        DEFAULT_CATALOG,
        load_song_catalog,
        load_user_history,
    )
    from models import ContentBasedRecommender


class MusicRecommendationService:
    """Load the catalog once and return JSON-friendly recommendation records."""

    def __init__(
        self,
        songs: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
        *,
        genre_weight: float = 1.0,
        text_weight: float = 1.0,
        lyrics_weight: float = 2.0,
    ) -> None:
        self.songs = songs.reset_index(drop=True).copy()
        self.interactions = interactions
        self.model = ContentBasedRecommender(
            genre_weight=genre_weight,
            text_weight=text_weight,
            lyrics_weight=lyrics_weight,
        ).fit(self.songs)
        self._track_to_song = dict(
            zip(self.songs["track_id"].astype(str), self.songs["song_id"])
        )

    @classmethod
    def from_files(
        cls,
        catalog_path: str | Path = DEFAULT_CATALOG,
        history_path: str | Path | None = None,
        **model_options: float,
    ) -> "MusicRecommendationService":
        songs = load_song_catalog(catalog_path)
        interactions = (
            load_user_history(history_path, songs) if history_path is not None else None
        )
        return cls(songs, interactions, **model_options)

    @classmethod
    def from_defaults(cls, **model_options: float) -> "MusicRecommendationService":
        """Load the curated catalog without synthetic user history."""
        return cls.from_files(**model_options)

    def recommend_for_tracks(
        self,
        track_ids: Iterable[object] | object,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
    ) -> list[dict[str, object]]:
        """Recommend from liked catalog IDs, suitable for a new app user."""
        seed_song_ids = self._lookup_song_ids(track_ids, "track_ids")
        excluded_song_ids = self._lookup_song_ids(
            exclude_track_ids, "exclude_track_ids", allow_none=True
        )
        results = self.model.recommend_from_items(
            seed_song_ids,
            top_k=top_k,
            exclude_song_ids=excluded_song_ids,
        )
        return self._records(results)

    def recommend_for_user(
        self,
        user_id: object,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
    ) -> list[dict[str, object]]:
        """Recommend from a user's positive listening history."""
        if self.interactions is None:
            raise ValueError("No interaction history was loaded for this service.")

        normalized_user_id = str(user_id).strip()
        user_history = self.interactions.loc[
            self.interactions["user_id"] == normalized_user_id
        ]
        if user_history.empty:
            raise ValueError(f"No listening history found for user_id={user_id!r}.")

        excluded_song_ids = set(user_history["song_id"].astype(int).tolist())
        excluded_song_ids.update(
            self._lookup_song_ids(
                exclude_track_ids, "exclude_track_ids", allow_none=True
            )
        )
        results = self.model.recommend(
            self.interactions,
            user_id=normalized_user_id,
            top_k=top_k,
            exclude_song_ids=sorted(excluded_song_ids),
        )
        return self._records(results)

    def _lookup_song_ids(
        self,
        track_ids: Iterable[object] | object | None,
        argument_name: str,
        *,
        allow_none: bool = False,
    ) -> list[int]:
        if track_ids is None:
            if allow_none:
                return []
            raise ValueError(f"{argument_name} must contain at least one track ID.")
        if isinstance(track_ids, (str, bytes)):
            values = [track_ids]
        else:
            try:
                values = list(track_ids)  # type: ignore[arg-type]
            except TypeError:
                values = [track_ids]

        normalized = [str(value).strip() for value in values]
        if not normalized and not allow_none:
            raise ValueError(f"{argument_name} must contain at least one track ID.")
        if any(not value for value in normalized):
            raise ValueError(f"{argument_name} cannot contain an empty track ID.")

        unknown = [track_id for track_id in normalized if track_id not in self._track_to_song]
        if unknown:
            raise ValueError(f"Unknown track ID(s): {unknown[:10]!r}")
        return [int(self._track_to_song[track_id]) for track_id in normalized]

    @staticmethod
    def _records(results: pd.DataFrame) -> list[dict[str, object]]:
        records = []
        for row in results.itertuples(index=False):
            record: dict[str, object] = {
                "track_id": str(row.track_id),
                "title": str(row.title),
                "artist": str(row.artist),
                "score": float(row.score),
            }
            for column in ("genre", "genre_confidence", "dataset_source"):
                if column in results.columns and pd.notna(getattr(row, column, None)):
                    record[column] = str(getattr(row, column))
            records.append(record)
        return records
