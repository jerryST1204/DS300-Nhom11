"""Load current content/popularity artifacts for app recommendations."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .models.emotion_aware import EmotionAwareRecommender
else:
    from models.emotion_aware import EmotionAwareRecommender


class TrainedModelBundle:
    """Runtime adapter for content similarity and catalog popularity.

    Collaborative filtering is intentionally not exposed here: the supplied
    interaction CSV is synthetic and cannot support claims about real users.
    """

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        with (self.artifacts_dir / "manifest.json").open(encoding="utf-8") as file:
            self.manifest = json.load(file)
        if self.manifest.get("format_version") != 2:
            raise ValueError(
                "Model artifacts are from the old synthetic-history pipeline. "
                "Rebuild them with `python -m ials_music_recommendation.train_model_artifacts`."
            )
        self._verify_artifacts()

        with (self.artifacts_dir / "mappings.json").open(encoding="utf-8") as file:
            self.mappings = json.load(file)
        self.content_model = joblib.load(self.artifacts_dir / "content_based.joblib")
        self.catalog = pd.read_csv(
            self.artifacts_dir / "catalog.csv", encoding="utf-8"
        ).set_index("song_id", drop=False)
        self.track_to_song = {
            str(track_id): int(song_id)
            for track_id, song_id in self.mappings["track_id_to_song_id"].items()
        }
        self.popularity_scores = self._metadata_popularity(self.catalog)
        self._emotion_model: EmotionAwareRecommender | None = None

    @classmethod
    def from_directory(
        cls, artifacts_dir: str | Path | None = None
    ) -> "TrainedModelBundle":
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "trained_artifacts"
        return cls(artifacts_dir)

    def recommend_popular(
        self,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
        user_id: object | None = None,
    ) -> list[dict[str, object]]:
        """Rank catalog tracks by available view and like counts."""
        top_k = self._top_k(top_k)
        if user_id is not None:
            raise ValueError(
                "Catalog popularity is not personalized. Use liked track IDs; "
                "the bundled user history is synthetic."
            )
        excluded = set(
            self._lookup_song_ids(
                exclude_track_ids, "exclude_track_ids", allow_none=True
            )
        )
        scores = self.popularity_scores.copy()
        if excluded:
            scores[list(excluded)] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")[:top_k]]
        return self._records(ranked, scores[ranked])

    def recommend_similar(
        self,
        track_ids: Iterable[object] | object,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
    ) -> list[dict[str, object]]:
        """Return tracks similar to one or more catalog track IDs."""
        seeds = self._lookup_song_ids(track_ids, "track_ids")
        excluded = self._lookup_song_ids(
            exclude_track_ids, "exclude_track_ids", allow_none=True
        )
        results = self.content_model.recommend_from_items(
            seeds,
            top_k=self._top_k(top_k),
            exclude_song_ids=excluded,
        )
        return self._records(
            results["song_id"].to_numpy(dtype=np.int64),
            results["score"].to_numpy(dtype=np.float64),
        )

    def recommend_emotion_aware(
        self,
        track_ids: Iterable[object] | object,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
        emotion_weight: float = 0.5,
    ) -> list[dict[str, object]]:
        """Blend content and emotion scores within the emotion artifact catalog."""
        top_k = self._top_k(top_k)
        if not np.isfinite(emotion_weight) or not 0 <= emotion_weight <= 1:
            raise ValueError("emotion_weight must be finite and between 0 and 1.")

        seed_song_ids = self._lookup_song_ids(track_ids, "track_ids")
        excluded_song_ids = self._lookup_song_ids(
            exclude_track_ids, "exclude_track_ids", allow_none=True
        )
        if self._emotion_model is None:
            emotion_artifacts_dir = self.artifacts_dir.parent / "emotion_artifacts"
            self._emotion_model = EmotionAwareRecommender.from_artifacts(
                self.catalog.reset_index(drop=True),
                emotion_artifacts_dir,
                emotion_weight=emotion_weight,
            )
        else:
            self._emotion_model.emotion_weight = float(emotion_weight)

        available_track_ids = set(
            self._emotion_model.songs["track_id"].astype(str)
        )
        seed_track_ids = [
            self.mappings["song_id_to_track_id"][int(song_id)]
            for song_id in seed_song_ids
        ]
        unsupported = [
            track_id
            for track_id in seed_track_ids
            if track_id not in available_track_ids
        ]
        if unsupported:
            raise ValueError(
                "Emotion-aware recommendations are available only for tracks "
                "included in emotion_artifacts. Unsupported track_id(s): "
                f"{unsupported[:10]!r}"
            )

        results = self._emotion_model.recommend_from_items(
            seed_song_ids,
            top_k=top_k,
            exclude_song_ids=excluded_song_ids,
        )
        records = self._records(
            results["song_id"].to_numpy(dtype=np.int64),
            results["score"].to_numpy(dtype=np.float64),
        )
        for record, row in zip(records, results.itertuples(index=False)):
            record["content_score"] = float(row.content_score)
            record["emotion_score"] = float(row.emotion_score)
            record["emotion_weight"] = float(emotion_weight)
        return records

    def recommend_new_user(
        self,
        liked_track_ids: Iterable[object] | object | None = None,
        top_k: int = 10,
        exclude_track_ids: Iterable[object] | object | None = None,
        content_weight: float = 0.7,
        preferred_genres: Iterable[object] | object | None = None,
        min_genre_confidence: str = "medium",
    ) -> list[dict[str, object]]:
        """Blend catalog popularity and liked-track content for a new user."""
        top_k = self._top_k(top_k)
        if not np.isfinite(content_weight) or not 0 <= content_weight <= 1:
            raise ValueError("content_weight must be between 0 and 1.")
        genres = self._normalize_ids(
            preferred_genres, "preferred_genres", allow_none=True
        )
        seed_ids = self._lookup_song_ids(
            liked_track_ids, "liked_track_ids", allow_none=True
        )
        excluded = self._lookup_song_ids(
            exclude_track_ids, "exclude_track_ids", allow_none=True
        )
        excluded_song_ids = list(
            dict.fromkeys([*excluded, *seed_ids])
        )
        excluded_track_ids = [
            self.mappings["song_id_to_track_id"][int(song_id)]
            for song_id in excluded_song_ids
        ]
        pool_size = len(self.catalog) if genres else min(max(top_k * 3, top_k), len(self.catalog))

        popular = self.recommend_popular(
            top_k=pool_size, exclude_track_ids=excluded_track_ids
        )
        popular = self._filter_genres(
            popular, genres, min_genre_confidence=min_genre_confidence
        )
        if not seed_ids or content_weight == 0:
            return popular[:top_k]

        seed_track_ids = [
            self.mappings["song_id_to_track_id"][int(song_id)] for song_id in seed_ids
        ]
        content = self.recommend_similar(
            seed_track_ids, top_k=pool_size, exclude_track_ids=excluded_track_ids
        )
        content = self._filter_genres(
            content, genres, min_genre_confidence=min_genre_confidence
        )

        fused: dict[int, float] = {}
        for weight, recommendations in (
            (1.0 - content_weight, popular),
            (content_weight, content),
        ):
            if weight == 0:
                continue
            for rank, recommendation in enumerate(recommendations, start=1):
                song_id = self.track_to_song[recommendation["track_id"]]
                fused[song_id] = fused.get(song_id, 0.0) + weight / (60.0 + rank)
        ranked = sorted(fused, key=lambda song_id: (-fused[song_id], song_id))[:top_k]
        return self._records(ranked, [fused[song_id] for song_id in ranked])

    def recommend_item_based(
        self, user_id: object, top_k: int = 10
    ) -> list[dict[str, object]]:
        del user_id, top_k
        raise ValueError(
            "Collaborative filtering is unavailable because the provided "
            "interaction data is synthetic."
        )

    def recommend_funk_svd(
        self, user_id: object, top_k: int = 10
    ) -> list[dict[str, object]]:
        del user_id, top_k
        raise ValueError(
            "Funk SVD is unavailable because the provided interaction data is synthetic."
        )

    def _filter_genres(
        self,
        recommendations: list[dict[str, object]],
        preferred_genres: list[str],
        *,
        min_genre_confidence: str,
    ) -> list[dict[str, object]]:
        if not preferred_genres:
            return recommendations
        if "genre" not in self.catalog.columns:
            raise ValueError("This catalog has no genre metadata.")

        available = {
            str(genre).strip().casefold(): str(genre).strip()
            for genre in self.catalog["genre"].dropna().unique()
            if str(genre).strip()
        }
        requested = {genre.casefold() for genre in preferred_genres}
        confidence_rank = {"low": 1, "medium": 2, "high": 3}
        minimum = min_genre_confidence.casefold()
        if minimum not in confidence_rank:
            raise ValueError("min_genre_confidence must be low, medium, or high.")
        unknown = sorted(genre for genre in requested if genre not in available)
        if unknown:
            raise ValueError(
                f"Unknown genre(s): {unknown!r}. Available genres: "
                f"{sorted(available.values())!r}"
            )
        return [
            item
            for item in recommendations
            if str(item.get("genre", "")).casefold() in requested
            and confidence_rank.get(
                str(item.get("genre_confidence", "low")).casefold(), 0
            )
            >= confidence_rank[minimum]
        ]

    def _lookup_song_ids(
        self,
        values: Iterable[object] | object | None,
        name: str,
        *,
        allow_none: bool = False,
    ) -> list[int]:
        track_ids = self._normalize_ids(values, name, allow_none=allow_none)
        unknown = [track_id for track_id in track_ids if track_id not in self.track_to_song]
        if unknown:
            raise ValueError(f"Unknown track ID(s): {unknown[:10]!r}")
        return [self.track_to_song[track_id] for track_id in track_ids]

    @staticmethod
    def _normalize_ids(
        values: Iterable[object] | object | None,
        name: str,
        *,
        allow_none: bool = False,
    ) -> list[str]:
        if values is None:
            if allow_none:
                return []
            raise ValueError(f"{name} must contain at least one track ID.")
        if isinstance(values, (str, bytes)):
            raw_values = [values]
        else:
            try:
                raw_values = list(values)  # type: ignore[arg-type]
            except TypeError:
                raw_values = [values]
        normalized = [str(value).strip() for value in raw_values]
        if not normalized and not allow_none:
            raise ValueError(f"{name} must contain at least one track ID.")
        if any(not value for value in normalized):
            raise ValueError(f"{name} cannot contain an empty track ID.")
        return normalized

    @staticmethod
    def _top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        return int(top_k)

    @staticmethod
    def _metadata_popularity(catalog: pd.DataFrame) -> np.ndarray:
        """Combine robust percentile ranks for view and like counts."""
        component_scores = []
        for column in ("view_count", "like_count"):
            if column not in catalog.columns:
                continue
            values = pd.to_numeric(catalog[column], errors="coerce")
            valid = values.notna() & values.ge(0)
            if not valid.any():
                continue
            ranks = pd.Series(np.nan, index=catalog.index, dtype=np.float64)
            if int(valid.sum()) == 1:
                ranks.loc[valid] = 1.0
            else:
                ranks.loc[valid] = (values.loc[valid].rank(method="average") - 1) / (
                    int(valid.sum()) - 1
                )
            component_scores.append(ranks.to_numpy(dtype=np.float64))
        if not component_scores:
            return np.full(len(catalog), -np.inf, dtype=np.float64)
        matrix = np.vstack(component_scores)
        valid = np.isfinite(matrix)
        counts = valid.sum(axis=0)
        scores = np.divide(
            np.nansum(matrix, axis=0),
            counts,
            out=np.full(len(catalog), -np.inf, dtype=np.float64),
            where=counts > 0,
        )
        return scores

    def _records(
        self, song_ids: Iterable[int], scores: Iterable[float]
    ) -> list[dict[str, object]]:
        records = []
        for song_id, score in zip(song_ids, scores):
            row = self.catalog.loc[int(song_id)]
            record: dict[str, object] = {
                "track_id": str(row["track_id"]),
                "title": str(row["title"]),
                "artist": str(row["artist"]),
                "score": float(score),
            }
            for column in ("genre", "genre_confidence", "dataset_source"):
                if column in row.index and pd.notna(row[column]):
                    record[column] = str(row[column])
            records.append(record)
        return records

    def _verify_artifacts(self) -> None:
        expected = self.manifest.get("sha256", {})
        if not expected:
            raise ValueError("Artifact manifest has no file checksums; retrain the model.")
        for name, expected_digest in expected.items():
            path = self.artifacts_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError(f"Artifact checksum mismatch: {name}; retrain the model.")
