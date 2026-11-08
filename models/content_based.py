"""Content-based recommendations from metadata and curated song lyrics."""

import html
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


_TIMESTAMP = re.compile(
    r"(?m)^\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\s+-->\s+.*$"
)
_HEADER = re.compile(r"(?im)^\s*(?:WEBVTT|Kind:.*|Language:.*)\s*$")
_TAG = re.compile(r"<[^>]*>")


def _clean_subtitle(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = _HEADER.sub(" ", text)
    text = _TIMESTAMP.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    text = re.sub(r"[♪♫]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tfidf_block(texts: pd.Series, n_rows: int) -> csr_matrix:
    values = texts.fillna("").astype(str).str.strip()
    if not values.ne("").any():
        return csr_matrix((n_rows, 0), dtype=np.float64)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=150_000,
        dtype=np.float64,
    )
    return vectorizer.fit_transform(values)


class ContentBasedRecommender:
    """Recommend unseen songs similar to a user's seed tracks or listening history."""

    def __init__(
        self,
        genre_weight: float = 1.0,
        text_weight: float = 1.0,
        lyrics_weight: float = 2.0,
    ):
        for name, value in (
            ("genre_weight", genre_weight),
            ("text_weight", text_weight),
            ("lyrics_weight", lyrics_weight),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")

        self.genre_weight = genre_weight
        self.text_weight = text_weight
        self.lyrics_weight = lyrics_weight
        self.songs: pd.DataFrame | None = None
        self.item_features: csr_matrix | None = None
        self._song_id_to_row: dict[object, int] = {}
        self._history_source: pd.DataFrame | None = None
        self._history_positions: dict[object, np.ndarray] = {}

    def fit(self, songs: pd.DataFrame) -> "ContentBasedRecommender":
        required = {"song_id", "title"}
        missing = required - set(songs.columns)
        if missing:
            raise ValueError(f"songs is missing columns: {', '.join(sorted(missing))}")
        if songs.empty:
            raise ValueError("songs must contain at least one row.")
        if songs[list(required)].isna().any().any():
            raise ValueError("song_id and title cannot contain missing values.")
        if songs["song_id"].duplicated().any():
            raise ValueError("song_id values must be unique.")

        catalog = songs.reset_index(drop=True).copy()
        catalog["title"] = catalog["title"].astype(str).str.strip()
        if catalog["title"].eq("").any():
            raise ValueError("title cannot be empty.")
        if "artist" not in catalog:
            catalog["artist"] = ""
        catalog["artist"] = catalog["artist"].fillna("").astype(str).str.strip()

        n_rows = len(catalog)
        feature_blocks = []
        if "genre" in catalog:
            genres = catalog["genre"].fillna("").astype(str).str.strip()
            valid = genres.ne("") & genres.ne("Other/Unknown")
            if valid.any():
                genre_codes, unique_genres = pd.factorize(genres[valid], sort=True)
                genre_rows = np.flatnonzero(valid.to_numpy())
                confidence = catalog.get(
                    "genre_confidence", pd.Series("unknown", index=catalog.index)
                ).fillna("unknown").astype(str).str.casefold()
                confidence_weights = confidence.map(
                    {
                        "high": 1.0,
                        "medium": 0.75,
                        "low": 0.25,
                        "very_low": 0.1,
                    }
                ).fillna(0.0)
                genre_features = csr_matrix(
                    (
                        confidence_weights.iloc[genre_rows].to_numpy(dtype=np.float64),
                        (genre_rows, genre_codes),
                    ),
                    shape=(n_rows, len(unique_genres)),
                )
                feature_blocks.append(genre_features * self.genre_weight)

        title_artist = catalog["title"] + " " + catalog["artist"]
        feature_blocks.append(_tfidf_block(title_artist, n_rows) * self.text_weight)

        lyrics_column = "lyrics_clean" if "lyrics_clean" in catalog else "lyrics"
        if lyrics_column in catalog:
            lyrics = catalog[lyrics_column].fillna("").astype(str).str.strip()
            if "lyrics_use_for_vector" in catalog:
                use_for_vector = (
                    catalog["lyrics_use_for_vector"]
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .str.casefold()
                    .isin({"true", "1", "yes", "y"})
                )
                lyrics = lyrics.where(use_for_vector, "")
            # Captions may contain speech or timestamps. Only curated lyrics
            # are used here; songs without approved lyrics rely on metadata.
            lyrics = lyrics.map(_clean_subtitle)
            feature_blocks.append(_tfidf_block(lyrics, n_rows) * self.lyrics_weight)

        self.item_features = normalize(hstack(feature_blocks, format="csr"), norm="l2")
        self.songs = catalog
        self._song_id_to_row = {
            song_id: row for row, song_id in enumerate(catalog["song_id"])
        }
        self._history_source = None
        self._history_positions = {}
        return self

    def recommend_from_items(
        self,
        seed_song_ids: list[object],
        top_k: int = 10,
        strengths: list[float] | None = None,
        exclude_song_ids: list[object] | None = None,
    ) -> pd.DataFrame:
        """Recommend from positive seeds, optionally excluding known songs."""
        if self.songs is None or self.item_features is None:
            raise ValueError("Call fit(songs) before requesting recommendations.")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
        if not seed_song_ids:
            raise ValueError("At least one seed song_id is required.")

        seed_rows = []
        for song_id in seed_song_ids:
            if song_id not in self._song_id_to_row:
                raise ValueError(f"Unknown song_id: {song_id!r}")
            seed_rows.append(self._song_id_to_row[song_id])
        if len(set(seed_rows)) != len(seed_rows):
            raise ValueError("seed_song_ids must not contain duplicates.")

        if strengths is None:
            weights = np.ones(len(seed_rows), dtype=np.float64)
        else:
            weights = np.asarray(strengths, dtype=np.float64)
            if weights.shape != (len(seed_rows),):
                raise ValueError("strengths must contain one weight per seed song.")
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("strengths must be finite and greater than zero.")

        profile = weights @ self.item_features[seed_rows]
        profile = normalize(csr_matrix(profile), norm="l2")
        scores = (profile @ self.item_features.T).toarray().ravel()
        scores[seed_rows] = -np.inf
        if exclude_song_ids is not None:
            unknown = [
                song_id
                for song_id in exclude_song_ids
                if song_id not in self._song_id_to_row
            ]
            if unknown:
                raise ValueError(f"Unknown song_id(s) to exclude: {unknown[:10]!r}")
            excluded_rows = [
                self._song_id_to_row[song_id] for song_id in exclude_song_ids
            ]
            scores[excluded_rows] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        ranked_rows = candidates[np.argsort(-scores[candidates], kind="stable")[:top_k]]

        recommendations = self.songs.iloc[ranked_rows].copy()
        recommendations.insert(0, "score", scores[ranked_rows])
        return recommendations.reset_index(drop=True)

    def recommend(
        self,
        interactions: pd.DataFrame,
        user_id: object,
        top_k: int = 10,
        exclude_song_ids: list[object] | None = None,
    ) -> pd.DataFrame:
        """Recommend from positive history while optionally excluding seen songs."""
        required = {"user_id", "song_id", "play_count"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(
                f"interactions is missing columns: {', '.join(sorted(missing))}"
            )

        # Reuse positional groups during batch evaluation instead of scanning
        # the full interaction table once for every user.
        if self._history_source is not interactions:
            self._history_positions = interactions.groupby(
                "user_id", sort=False
            ).indices
            self._history_source = interactions
        positions = self._history_positions.get(user_id)
        if positions is None or len(positions) == 0:
            raise ValueError(f"No listening history found for user_id={user_id!r}.")
        history = interactions.iloc[positions].copy()
        if history[["song_id", "play_count"]].isna().any().any():
            raise ValueError("song_id and play_count cannot contain missing values.")
        history["play_count"] = pd.to_numeric(history["play_count"], errors="raise")
        if (
            not np.isfinite(history["play_count"]).all()
            or (history["play_count"] < 0).any()
        ):
            raise ValueError("play_count must contain finite, non-negative values.")

        history = history.loc[history["play_count"] > 0]
        if "rating" in history.columns:
            history["rating"] = pd.to_numeric(history["rating"], errors="raise")
            if (
                not np.isfinite(history["rating"]).all()
                or (history["rating"] < 1).any()
                or (history["rating"] > 5).any()
            ):
                raise ValueError("rating must be finite and within [1, 5].")
            history = history.loc[history["rating"] >= 4]
        if history.empty:
            raise ValueError(
                f"No positive or well-rated listening history found for {user_id!r}."
            )
        aggregations = {"play_count": "sum"}
        if "rating" in history.columns:
            aggregations["rating"] = "mean"
        history = history.groupby("song_id", as_index=False).agg(aggregations)
        seed_ids = history["song_id"].tolist()
        weights = np.log1p(history["play_count"].to_numpy(dtype=np.float64))
        if "rating" in history.columns:
            weights *= (history["rating"].to_numpy(dtype=np.float64) - 3.0)
        return self.recommend_from_items(
            seed_ids,
            top_k=top_k,
            strengths=weights,
            exclude_song_ids=exclude_song_ids,
        )
