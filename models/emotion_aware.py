"""Experimental recommender blending local emotion embeddings with TF-IDF."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .content_based import ContentBasedRecommender


class EmotionAwareRecommender:
    """Blend baseline content scores with similarities between emotion vectors."""

    def __init__(
        self,
        songs: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_track_ids: list[str],
        emotion_weight: float = 0.5,
    ) -> None:
        if not 0 <= emotion_weight <= 1 or not np.isfinite(emotion_weight):
            raise ValueError("emotion_weight must be finite and between 0 and 1.")
        if "track_id" not in songs or "song_id" not in songs:
            raise ValueError("songs must contain track_id and song_id columns.")
        if songs["track_id"].astype(str).duplicated().any():
            raise ValueError("songs track_id values must be unique.")
        if len(embedding_track_ids) != len(set(embedding_track_ids)):
            raise ValueError("embedding track IDs must be unique.")
        if set(songs["track_id"].astype(str)) != set(embedding_track_ids):
            raise ValueError("Embedding track IDs do not match the song catalog.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(embedding_track_ids):
            raise ValueError("Embedding rows must match the embedding track IDs.")
        if not np.isfinite(matrix).all():
            raise ValueError("Embeddings must contain only finite values.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Embedding vectors cannot be zero.")
        matrix = matrix / norms

        row_by_track_id = {
            track_id: row for row, track_id in enumerate(embedding_track_ids)
        }
        order = [row_by_track_id[str(track_id)] for track_id in songs["track_id"]]
        self.embeddings = matrix[order]
        self.songs = songs.reset_index(drop=True).copy()
        self.emotion_weight = float(emotion_weight)
        self._row_by_song_id = {
            song_id: row for row, song_id in enumerate(self.songs["song_id"])
        }
        self.baseline = ContentBasedRecommender().fit(self.songs)

    @classmethod
    def from_artifacts(
        cls,
        songs: pd.DataFrame,
        artifact_dir: str | Path,
        emotion_weight: float = 0.5,
    ) -> "EmotionAwareRecommender":
        artifact_dir = Path(artifact_dir)
        matrix_path = artifact_dir / "vietnam_song_emotion_embeddings.npy"
        ids_path = artifact_dir / "song_ids.json"
        manifest_path = artifact_dir / "manifest.json"
        sample_only = False
        if not matrix_path.is_file() or not ids_path.is_file():
            matrix_path = artifact_dir / "vietnam_song_emotion_embeddings.sample.npy"
            ids_path = artifact_dir / "song_ids.sample.json"
            manifest_path = artifact_dir / "manifest.sample.json"
            sample_only = True
        if not matrix_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                f"Emotion artifacts not found in {artifact_dir}. "
                "Run emotion_pipeline.py first."
            )
        embeddings = np.load(matrix_path, allow_pickle=False)
        with ids_path.open(encoding="utf-8") as file:
            track_ids = json.load(file)
        if not isinstance(track_ids, list) or not all(
            isinstance(track_id, str) for track_id in track_ids
        ):
            raise ValueError("song_ids.json must contain a list of string track IDs.")
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            sample_only = bool(manifest.get("sample_only", sample_only))
        if sample_only:
            included_ids = set(track_ids)
            songs = songs.loc[songs["track_id"].astype(str).isin(included_ids)].copy()
            if len(songs) != len(included_ids):
                raise ValueError("Sample artifact IDs do not match the song catalog.")
        return cls(songs, embeddings, track_ids, emotion_weight)

    def recommend_from_items(
        self,
        seed_song_ids: list[object],
        top_k: int = 10,
        strengths: list[float] | None = None,
        exclude_song_ids: list[object] | None = None,
    ) -> pd.DataFrame:
        """Return recommendations ranked by a weighted baseline/emotion score."""
        return self.recommend_from_items_for_weights(
            seed_song_ids,
            [self.emotion_weight],
            top_k=top_k,
            strengths=strengths,
            exclude_song_ids=exclude_song_ids,
        )[self.emotion_weight]

    def recommend_from_items_for_weights(
        self,
        seed_song_ids: list[object],
        emotion_weights: list[float],
        top_k: int = 10,
        strengths: list[float] | None = None,
        exclude_song_ids: list[object] | None = None,
    ) -> dict[float, pd.DataFrame]:
        """Rank once per seed and reuse component scores for several blends."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
        if not seed_song_ids:
            raise ValueError("At least one seed song_id is required.")
        if not emotion_weights:
            raise ValueError("At least one emotion weight is required.")
        weights_to_score = [float(weight) for weight in emotion_weights]
        if (
            len(weights_to_score) != len(set(weights_to_score))
            or any(
                not np.isfinite(weight) or not 0 <= weight <= 1
                for weight in weights_to_score
            )
        ):
            raise ValueError(
                "emotion_weights must be unique finite values between 0 and 1."
            )

        try:
            seed_rows = [self._row_by_song_id[song_id] for song_id in seed_song_ids]
        except KeyError as error:
            raise ValueError(f"Unknown song_id: {error.args[0]!r}") from error
        if len(seed_rows) != len(set(seed_rows)):
            raise ValueError("seed_song_ids must not contain duplicates.")

        if strengths is None:
            weights = np.ones(len(seed_rows), dtype=np.float32)
        else:
            weights = np.asarray(strengths, dtype=np.float32)
            if weights.shape != (len(seed_rows),):
                raise ValueError("strengths must contain one weight per seed song.")
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("strengths must be finite and greater than zero.")

        profile = weights @ self.embeddings[seed_rows]
        profile_norm = np.linalg.norm(profile)
        if not np.isfinite(profile_norm) or profile_norm == 0:
            raise ValueError("Could not construct a valid emotion profile.")
        profile /= profile_norm

        excluded = set(seed_song_ids)
        if exclude_song_ids is not None:
            excluded.update(exclude_song_ids)
        for song_id in excluded:
            if song_id not in self._row_by_song_id:
                raise ValueError(f"Unknown song_id to exclude: {song_id!r}")

        baseline_results = self.baseline.recommend_from_items(
            seed_song_ids,
            top_k=len(self.songs),
            strengths=None if strengths is None else list(weights.astype(float)),
            exclude_song_ids=list(excluded - set(seed_song_ids)),
        )
        baseline_scores = np.full(len(self.songs), -np.inf, dtype=np.float32)
        for row in baseline_results.itertuples(index=False):
            song_id = getattr(row, "song_id")
            baseline_scores[self._row_by_song_id[song_id]] = float(row.score)

        emotion_scores = np.clip(self.embeddings @ profile, -1.0, 1.0)
        emotion_scores = (emotion_scores + 1.0) / 2.0
        available = np.isfinite(baseline_scores)
        excluded_rows = [
            self._row_by_song_id[song_id] for song_id in excluded
        ]
        candidates = np.flatnonzero(available)
        results: dict[float, pd.DataFrame] = {}
        for emotion_weight in weights_to_score:
            combined = np.full(len(self.songs), -np.inf, dtype=np.float32)
            combined[available] = (
                (1.0 - emotion_weight) * baseline_scores[available]
                + emotion_weight * emotion_scores[available]
            )
            combined[excluded_rows] = -np.inf
            valid_candidates = candidates[np.isfinite(combined[candidates])]
            ranked_rows = valid_candidates[
                np.argsort(-combined[valid_candidates], kind="stable")[:top_k]
            ]
            recommendations = self.songs.iloc[ranked_rows].copy()
            recommendations.insert(0, "score", combined[ranked_rows])
            recommendations.insert(1, "content_score", baseline_scores[ranked_rows])
            recommendations.insert(2, "emotion_score", emotion_scores[ranked_rows])
            results[emotion_weight] = recommendations.reset_index(drop=True)
        return results

    def recommend_similar(
        self, track_id: str, top_k: int = 10
    ) -> pd.DataFrame:
        """Recommend songs similar in content and summarized emotional meaning."""
        matches = self.songs.index[self.songs["track_id"].astype(str).eq(str(track_id))]
        if len(matches) == 0:
            raise ValueError(f"Unknown track_id: {track_id!r}")
        song_id = self.songs.iloc[int(matches[0])]["song_id"]
        return self.recommend_from_items([song_id], top_k=top_k)
