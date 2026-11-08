"""Load the curated music catalog and optional listening-history examples."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent / "Dataset"
DEFAULT_CATALOG = DATASET_DIR / "data" / "lyrics_all_sources_cleaned.csv"
DEFAULT_USER_HISTORY = DATASET_DIR / "user data" / "synthetic_user_song_data.csv"
DEFAULT_VIETNAMESE_LYRICS = (
    DATASET_DIR / "dataViet" / "tracks_with_lyrics_vector_ready.csv"
)
DEFAULT_VMUSIC_LYRICS = (
    DATASET_DIR / "data" / "VMusic_lyrics_V3_2_vector_ready.csv"
)


def _text_column(raw: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """Return the first non-empty text value from a list of possible columns."""
    result = pd.Series("", index=raw.index, dtype="string")
    for column in candidates:
        if column not in raw:
            continue
        values = raw[column].astype("string").fillna("").str.strip()
        result = result.mask(result.eq(""), values)
    return result.fillna("")


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def load_song_catalog(csv_path: str | Path = DEFAULT_CATALOG) -> pd.DataFrame:
    """Load and normalize the current combined VMusic + dataViet export.

    The full cleaned export keeps every known track in the catalog, but lyrics
    enter the model only when the data manager marked them
    ``lyrics_use_for_vector``. This preserves tracks without verified lyrics
    while keeping unreviewed lyric text out of the feature vectors.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy catalog nhạc: {csv_path}")

    raw = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    if not {"title", "artist"}.issubset(raw.columns):
        raise ValueError("Catalog cần có các cột title và artist.")
    if not ({"id", "track_id"} & set(raw.columns)):
        raise ValueError("Catalog cần có cột id hoặc track_id.")

    source_ids = _text_column(raw, ("id", "track_id"))
    songs = pd.DataFrame(
        {
            "track_id": source_ids,
            "title": _text_column(raw, ("title_clean", "title")),
            "artist": _text_column(raw, ("artist_clean", "artist")),
        }
    )

    lyrics = _text_column(raw, ("lyrics_clean", "lyrics"))
    if "lyrics_use_for_vector" in raw:
        lyrics = lyrics.where(_truthy(raw["lyrics_use_for_vector"]), "")
    songs["lyrics"] = lyrics

    copy_columns = (
        "dataset_source",
        "genre",
        "genre_confidence",
        "genre_source",
        "lyrics_quality",
        "lyrics_source",
        "lyrics_status",
        "match_score",
        "subtitle_clean",
        "duration",
        "duration_seconds",
        "view_count",
        "like_count",
        "upload_date",
        "lyrics_use_for_vector",
        "lyrics_clean_is_duplicate",
    )
    for column in copy_columns:
        if column in raw:
            songs[column] = raw[column]

    # Older exports may carry duration under either source-specific name.
    if "duration" not in songs and "duration_seconds" in songs:
        songs["duration"] = songs["duration_seconds"]
    elif "duration" in songs and "duration_seconds" in songs:
        songs["duration"] = songs["duration"].fillna(songs["duration_seconds"])

    songs["track_id"] = songs["track_id"].fillna("").astype(str).str.strip()
    songs["title"] = songs["title"].fillna("").astype(str).str.strip()
    songs["artist"] = songs["artist"].fillna("").astype(str).str.strip()
    songs["lyrics"] = songs["lyrics"].fillna("").astype(str).str.strip()
    songs["subtitle_clean"] = (
        songs.get("subtitle_clean", pd.Series("", index=songs.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_rows = songs["track_id"].ne("") & songs["title"].ne("")
    invalid_count = int((~valid_rows).sum())
    if invalid_count:
        warnings.warn(
            f"Skipping {invalid_count} catalog row(s) without a valid track_id or title.",
            UserWarning,
            stacklevel=2,
        )
    songs = songs.loc[valid_rows].copy()
    if songs.empty:
        raise ValueError("Catalog không có bài hát với track_id và title hợp lệ.")
    if songs["track_id"].duplicated().any():
        duplicates = songs.loc[songs["track_id"].duplicated(), "track_id"].head(10)
        raise ValueError(
            "track_id phải duy nhất; mã bị trùng: "
            f"{duplicates.astype(str).tolist()!r}"
        )

    songs["song_id"] = np.arange(len(songs), dtype=np.int64)
    songs["has_lyrics"] = songs["lyrics"].ne("")
    songs["has_subtitle"] = songs["subtitle_clean"].ne("")
    for column in ("duration", "view_count", "like_count", "match_score"):
        if column in songs:
            songs[column] = pd.to_numeric(songs[column], errors="coerce")
    if "artist" not in songs:
        songs["artist"] = ""
    return songs.reset_index(drop=True)


def load_user_history(
    csv_path: str | Path,
    songs: pd.DataFrame,
) -> pd.DataFrame:
    """Load interactions; the bundled history is synthetic and demo-only.

    Each row is one listening event unless ``play_count`` is supplied.
    Repeated user/song rows are combined. Unknown track IDs are rejected so
    that a mismatched catalog cannot silently corrupt an evaluation.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy lịch sử nghe: {csv_path}")

    history = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    history.columns = [
        str(column).strip().lower().replace(" ", "_") for column in history.columns
    ]
    aliases = {
        "user_id": ("user", "userid"),
        "track_id": ("song_id", "song", "songid", "trackid"),
        "play_count": ("playcount", "plays"),
    }
    for canonical, alternatives in aliases.items():
        if canonical not in history:
            match = next((name for name in alternatives if name in history), None)
            if match is not None:
                history = history.rename(columns={match: canonical})
    if "user_id" not in history.columns:
        raise ValueError("Lịch sử nghe cần có cột user_id.")
    if "track_id" in history.columns:
        track_to_song = dict(zip(songs["track_id"].astype(str), songs["song_id"]))
        history["track_id"] = history["track_id"].astype("string").str.strip()
        unknown = set(history["track_id"].dropna().astype(str)) - set(track_to_song)
        if unknown:
            raise ValueError(
                "Một số track_id trong lịch sử không có trong catalog: "
                f"{sorted(unknown)[:10]!r}"
            )
        history["song_id"] = history["track_id"].map(track_to_song)
    elif "song_id" in history.columns:
        history["song_id"] = pd.to_numeric(history["song_id"], errors="raise")
        if (
            not np.isfinite(history["song_id"]).all()
            or (history["song_id"] % 1 != 0).any()
        ):
            raise ValueError("song_id phải là số nguyên.")
        history["song_id"] = history["song_id"].astype(np.int64)
        unknown = set(history["song_id"]) - set(songs["song_id"])
        if unknown:
            raise ValueError(f"song_id không có trong catalog: {sorted(unknown)[:10]}")
    else:
        raise ValueError("Lịch sử nghe cần có cột track_id hoặc song_id.")

    if history["user_id"].isna().any() or history["song_id"].isna().any():
        raise ValueError("user_id và mã bài hát không được để trống.")
    history["user_id"] = history["user_id"].astype(str).str.strip()
    if history["user_id"].eq("").any():
        raise ValueError("user_id không được để trống.")

    if "play_count" in history.columns:
        history["play_count"] = pd.to_numeric(history["play_count"], errors="raise")
    else:
        history["play_count"] = 1
    if (
        not np.isfinite(history["play_count"]).all()
        or (history["play_count"] <= 0).any()
    ):
        raise ValueError("play_count phải hữu hạn và lớn hơn 0.")

    if "rating" in history.columns:
        history["rating"] = pd.to_numeric(history["rating"], errors="raise")
        if (
            not np.isfinite(history["rating"]).all()
            or (history["rating"] < 1).any()
            or (history["rating"] > 5).any()
        ):
            raise ValueError("rating phải nằm trong khoảng [1, 5].")
        aggregations = {"play_count": "sum", "rating": "mean"}
    else:
        aggregations = {"play_count": "sum"}

    return (
        history.groupby(["user_id", "song_id"], as_index=False)
        .agg(aggregations)
        .sort_values(["user_id", "song_id"], kind="stable")
        .reset_index(drop=True)
    )


def generate_synthetic_interactions(
    songs: pd.DataFrame,
    num_users: int = 300,
    interactions_per_user: int = 20,
    num_communities: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate demo-only pseudo-listening; never treat it as user evidence."""
    if songs.empty:
        raise ValueError("Catalog cần có ít nhất một bài hát.")
    if num_users < 1 or interactions_per_user < 2 or num_communities < 1:
        raise ValueError("Số user, lượt nghe và cộng đồng phải hợp lệ.")
    if interactions_per_user > len(songs):
        raise ValueError("interactions_per_user lớn hơn số bài trong catalog.")
    if not {"song_id", "title", "artist"}.issubset(songs.columns):
        raise ValueError("Catalog cần có song_id, title và artist.")

    if __package__:
        from .models import ContentBasedRecommender
    else:
        from models import ContentBasedRecommender

    rng = np.random.default_rng(random_state)
    features = ContentBasedRecommender().fit(songs).item_features
    song_ids = songs["song_id"].to_numpy()
    artists = songs["artist"].fillna("").astype(str).str.strip()
    artist_sizes = artists.loc[artists.ne("")].value_counts()
    eligible_artists = artist_sizes[artist_sizes >= 3]
    if eligible_artists.empty:
        raise ValueError("Cần nghệ sĩ có ít nhất 3 bài để tạo dữ liệu demo.")

    community_count = min(num_communities, len(eligible_artists))
    artist_names = eligible_artists.index.to_numpy()
    artist_weights = np.sqrt(eligible_artists.to_numpy(dtype=np.float64))
    artist_weights /= artist_weights.sum()
    community_artists = rng.choice(
        artist_names, size=community_count, replace=False, p=artist_weights
    )
    pool_size = min(max(interactions_per_user * 2, 50), len(songs))
    profiles: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    for artist in community_artists:
        artist_rows = np.flatnonzero(artists.to_numpy() == artist)
        seed_row = int(rng.choice(artist_rows))
        similarities = (features[seed_row] @ features.T).toarray().ravel()
        similarities[seed_row] = -np.inf
        candidates = np.flatnonzero(np.isfinite(similarities) & (similarities > 0))
        if len(candidates) > pool_size - 1:
            order = np.argsort(-similarities[candidates], kind="stable")
            candidates = candidates[order[: pool_size - 1]]
        if len(candidates) < interactions_per_user - 1:
            remaining = np.setdiff1d(
                np.arange(len(songs)), np.concatenate(([seed_row], candidates))
            )
            extra_count = min(interactions_per_user - 1 - len(candidates), len(remaining))
            if extra_count:
                candidates = np.concatenate(
                    [candidates, rng.choice(remaining, size=extra_count, replace=False)]
                )
        weights = np.maximum(similarities[candidates], 0.0) ** 3 + 1e-12
        profiles[str(artist)] = (
            seed_row,
            candidates,
            weights / weights.sum(),
            similarities,
        )

    rows = []
    for user_number in range(num_users):
        artist = str(community_artists[user_number % community_count])
        seed_row, candidates, probabilities, similarities = profiles[artist]
        selected = rng.choice(
            candidates, size=interactions_per_user - 1, replace=False, p=probabilities
        )
        for row in np.concatenate(([seed_row], selected)):
            affinity = max(float(similarities[row]), 0.0)
            play_count = int(rng.integers(2, 6)) if row == seed_row else int(
                np.clip(1 + round(affinity * 8), 1, 8)
            )
            rows.append((f"synthetic_{user_number:04d}", song_ids[row], play_count))

    interactions = pd.DataFrame(rows, columns=["user_id", "song_id", "play_count"])
    return interactions.sort_values(
        ["user_id", "song_id"], kind="stable"
    ).reset_index(drop=True)
