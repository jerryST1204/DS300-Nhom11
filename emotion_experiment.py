"""Compare emotion-aware ranking weights on approved music catalog tracks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __package__:
    from .data_generator import (
        DEFAULT_USER_HISTORY,
        load_song_catalog,
        load_user_history,
    )
    from .evaluator import intra_list_diversity_at_k, long_tail_share_at_k
    from .models.emotion_aware import EmotionAwareRecommender
else:
    from data_generator import DEFAULT_USER_HISTORY, load_song_catalog, load_user_history
    from evaluator import intra_list_diversity_at_k, long_tail_share_at_k
    from models.emotion_aware import EmotionAwareRecommender


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "emotion_artifacts"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "experiment"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOP_K = 10
RANDOM_STATE = 300


def _approved_dataviet(songs: pd.DataFrame) -> pd.DataFrame:
    if "dataset_source" not in songs:
        raise ValueError("Catalog is missing dataset_source.")
    source = songs["dataset_source"].fillna("").astype(str).str.strip().str.casefold()
    approved = songs["lyrics"].fillna("").astype(str).str.strip().ne("")
    result = songs.loc[source.eq("dataviet") & approved].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("No approved dataViet lyrics are available.")
    result["song_id"] = np.arange(len(result), dtype=np.int64)
    return result


def _stratified_tracks(
    songs: pd.DataFrame, per_genre: int, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pieces = []
    genres = songs["genre"].fillna("Unknown").astype(str)
    for genre in sorted(genres.unique()):
        group = songs.loc[genres.eq(genre)]
        count = min(per_genre, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        pieces.append(songs.loc[chosen])
    return pd.concat(pieces).sort_values(["genre", "track_id"]).reset_index(drop=True)


def _load_sample_model(
    songs: pd.DataFrame,
    artifact_dir: Path,
    emotion_weight: float,
) -> EmotionAwareRecommender:
    model = EmotionAwareRecommender.from_artifacts(
        songs, artifact_dir, emotion_weight=emotion_weight
    )
    if len(model.songs) != len(songs):
        raise ValueError(
            "The emotion artifacts do not cover every approved dataViet track. "
            "Run emotion_pipeline.py with --max-items 335 --dataset-source dataViet."
        )
    return model


def _local_popularity(
    songs: pd.DataFrame, catalog: pd.DataFrame, history_path: Path
) -> tuple[np.ndarray, str]:
    if "view_count" in catalog.columns:
        views = pd.to_numeric(catalog["view_count"], errors="coerce").fillna(0.0)
        views_by_track = dict(
            zip(catalog["track_id"].astype(str), views.to_numpy(dtype=np.float64))
        )
        view_popularity = np.array(
            [views_by_track.get(str(track_id), 0.0) for track_id in songs["track_id"]],
            dtype=np.float64,
        )
        if np.count_nonzero(view_popularity) and np.unique(view_popularity).size > 1:
            return view_popularity, "catalog view_count metadata"

    history = load_user_history(history_path, catalog)
    totals = history.groupby("song_id")["play_count"].sum()
    global_to_local = dict(zip(songs["track_id"].astype(str), songs["song_id"]))
    catalog_track_ids = catalog["track_id"].astype(str)
    catalog_song_ids = catalog["song_id"].to_numpy()
    popularity = np.zeros(len(songs), dtype=np.float64)
    for track_id, global_song_id in zip(catalog_track_ids, catalog_song_ids):
        local_song_id = global_to_local.get(track_id)
        if local_song_id is not None:
            popularity[local_song_id] = float(totals.get(global_song_id, 0.0))
    if np.count_nonzero(popularity) and np.unique(popularity).size > 1:
        return popularity, "bundled synthetic listening history"
    return popularity, "unavailable (no varying popularity signal for dataViet)"


def _rank_for_seed(
    model: EmotionAwareRecommender, song_id: int, weight: float
) -> pd.DataFrame:
    model.emotion_weight = weight
    return model.recommend_from_items([song_id], top_k=TOP_K)


def run_experiment(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    history_path: str | Path = DEFAULT_USER_HISTORY,
    random_state: int = RANDOM_STATE,
    human_seed_count_per_genre: int = 4,
    summary_review_count_per_genre: int = 6,
) -> dict[str, object]:
    """Write weight metrics and blinded human-review sheets."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_song_catalog()
    songs = _approved_dataviet(catalog)
    model = _load_sample_model(songs, artifact_dir, emotion_weight=0.5)
    if "genre" not in songs:
        raise ValueError("Catalog is missing genre.")
    popularity, popularity_source = _local_popularity(
        songs, catalog, Path(history_path)
    )
    long_tail_available = popularity_source.startswith(
        ("catalog view_count metadata", "bundled synthetic listening history")
    )

    comparison_seeds = songs.sort_values(["genre", "track_id"]).reset_index(drop=True)
    per_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    baseline_rankings: dict[int, list[str]] = {}
    rankings_by_weight: dict[float, dict[int, pd.DataFrame]] = {
        weight: {} for weight in WEIGHTS
    }

    for seed in comparison_seeds.itertuples(index=False):
        seed_song_id = int(seed.song_id)
        baseline = _rank_for_seed(model, seed_song_id, 0.0)
        baseline_ids = baseline["track_id"].astype(str).tolist()
        baseline_rankings[seed_song_id] = baseline_ids
        for weight in WEIGHTS:
            results = _rank_for_seed(model, seed_song_id, weight)
            rankings_by_weight[weight][seed_song_id] = results
            recommended = results["song_id"].to_numpy(dtype=np.int64)
            top = results.head(TOP_K)
            overlap = len(
                set(top["track_id"].astype(str)) & set(baseline_ids[:TOP_K])
            ) / TOP_K
            per_seed_rows.append(
                {
                    "seed_track_id": str(seed.track_id),
                    "seed_title": str(seed.title),
                    "seed_genre": str(seed.genre),
                    "emotion_weight": weight,
                    "baseline_top10_overlap": overlap,
                    "ild_at_10": intra_list_diversity_at_k(
                        recommended, model.baseline.item_features, TOP_K
                    ),
                    "long_tail_share_at_10": (
                        long_tail_share_at_k(recommended, popularity, TOP_K)
                        if long_tail_available else np.nan
                    ),
                    "mean_emotion_similarity": float(top["emotion_score"].mean()),
                    "unique_artists_at_10": int(
                        top["artist"].fillna("").astype(str).nunique()
                    ),
                }
            )
        if (seed_song_id + 1) % 50 == 0:
            print(f"[COMPARE] {seed_song_id + 1}/{len(songs)} seeds", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    for weight, group in per_seed.groupby("emotion_weight", sort=True):
        summary_rows.append(
            {
                "emotion_weight": float(weight),
                "n_seeds": int(group["seed_track_id"].nunique()),
                "baseline_top10_overlap_mean": float(
                    group["baseline_top10_overlap"].mean()
                ),
                "ild_at_10_mean": float(group["ild_at_10"].mean()),
                "long_tail_share_at_10_mean": float(
                    group["long_tail_share_at_10"].mean()
                ),
                "mean_emotion_similarity": float(
                    group["mean_emotion_similarity"].mean()
                ),
                "unique_artists_at_10_mean": float(
                    group["unique_artists_at_10"].mean()
                ),
            }
        )

    summary_path = output_dir / "weight_comparison.csv"
    per_seed_path = output_dir / "weight_comparison_by_seed.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    per_seed.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    blind_seeds = _stratified_tracks(
        songs, human_seed_count_per_genre, random_state
    )
    rng = np.random.default_rng(random_state)
    blind_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    codes = np.array(list("ABCDE"))
    for seed in blind_seeds.itertuples(index=False):
        assigned_codes = rng.permutation(codes).tolist()
        seed_song_id = int(seed.song_id)
        for weight, code in zip(WEIGHTS, assigned_codes):
            results = rankings_by_weight[weight][seed_song_id].head(TOP_K)
            recommendations = [
                {
                    "rank": rank,
                    "track_id": str(row.track_id),
                    "title": str(row.title),
                    "artist": str(row.artist),
                    "genre": str(row.genre),
                }
                for rank, row in enumerate(results.itertuples(index=False), start=1)
            ]
            blind_rows.append(
                {
                    "seed_track_id": str(seed.track_id),
                    "seed_title": str(seed.title),
                    "seed_artist": str(seed.artist),
                    "seed_genre": str(seed.genre),
                    "list_code": code,
                    "recommendations": json.dumps(
                        recommendations, ensure_ascii=False
                    ),
                    "mood_fit_1to5": "",
                    "relevance_1to5": "",
                    "variety_1to5": "",
                    "would_use": "",
                    "reviewer_notes": "",
                }
            )
            key_rows.append(
                {
                    "seed_track_id": str(seed.track_id),
                    "list_code": code,
                    "emotion_weight": weight,
                    "system": (
                        "Content-Based baseline"
                        if weight == 0
                        else "Emotion-aware blend"
                    ),
                }
            )
    blind_path = output_dir / "blind_review_lists.csv"
    blind_key_path = output_dir / "blind_review_key.csv"
    pd.DataFrame(blind_rows).to_csv(blind_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(key_rows).to_csv(blind_key_path, index=False, encoding="utf-8-sig")

    review_tracks = _stratified_tracks(
        songs, summary_review_count_per_genre, random_state + 1
    )
    summaries_path = artifact_dir / "emotion_summaries.csv"
    summaries = pd.read_csv(summaries_path, encoding="utf-8-sig")
    if "summary_status" not in summaries:
        summaries["summary_status"] = "ok"
    fallback_ids = set(
        summaries.loc[
            summaries["summary_status"].eq("neutral_fallback"), "track_id"
        ].astype(str)
    )
    selected_ids = set(review_tracks["track_id"].astype(str))
    fallback_tracks = songs.loc[
        songs["track_id"].astype(str).isin(fallback_ids - selected_ids)
    ]
    review_tracks = pd.concat([review_tracks, fallback_tracks], ignore_index=True)
    review = review_tracks.merge(
        summaries[["track_id", "emotion_summary", "summary_status"]],
        on="track_id",
        how="left",
        validate="one_to_one",
    )
    if review["emotion_summary"].isna().any():
        raise ValueError("Missing emotion summaries for the manual review sample.")
    summary_review_path = output_dir / "summary_quality_review.csv"
    review[
        [
            "track_id", "title", "artist", "genre",
            "emotion_summary", "summary_status",
        ]
    ].assign(
        summary_accurate_1to5="",
        mood_match_1to5="",
        needs_correction="",
        reviewer_notes="",
    ).to_csv(summary_review_path, index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": "dataViet",
        "approved_lyrics_tracks": int(len(songs)),
        "genre_counts": {
            str(key): int(value)
            for key, value in songs["genre"].fillna("Unknown").value_counts().items()
        },
        "emotion_weights": list(WEIGHTS),
        "top_k": TOP_K,
        "comparison_seed_count": int(len(comparison_seeds)),
        "human_blind_seed_count": int(len(blind_seeds)),
        "manual_summary_review_count": int(len(review)),
        "random_state": random_state,
        "popularity_source": popularity_source,
        "popularity_warning": (
            "LongTailShare is unavailable when all dataViet tracks have zero "
            "or identical popularity. If using bundled interactions as a "
            "fallback, they are synthetic and are not real listener popularity."
            if not long_tail_available
            or popularity_source == "bundled synthetic listening history"
            else ""
        ),
        "accuracy_warning": (
            "No emotion labels or real listening ground truth are available. "
            "These metrics compare ranking diversity/tail behavior, not accuracy."
        ),
        "files": [
            summary_path.name,
            per_seed_path.name,
            blind_path.name,
            blind_key_path.name,
            summary_review_path.name,
        ],
    }
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Compared {len(WEIGHTS)} weights across {len(songs)} dataViet seeds.")
    if not long_tail_available:
        print(
            "[WARN] LongTailShare is unavailable: dataViet has no varying "
            "view-count or interaction popularity signal.",
            flush=True,
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Blind recommendation lists: {blind_path.name}")
    print(f"Keep this key private until ratings are complete: {blind_key_path.name}")
    print(f"Summary review sheet: {summary_review_path.name}")
    return manifest


def run_full_catalog_experiment(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "full_catalog",
) -> dict[str, object]:
    """Compare all weights across every track in the full emotion artifact."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_song_catalog().reset_index(drop=True)
    model = EmotionAwareRecommender.from_artifacts(
        catalog, artifact_dir, emotion_weight=0.5
    )
    if len(model.songs) != len(catalog):
        raise ValueError(
            "Full emotion artifacts must cover the complete recommendation catalog."
        )
    if "lyrics" not in catalog:
        raise ValueError("Catalog is missing the approved lyrics field.")

    summary_path = artifact_dir / "emotion_summaries.csv"
    summaries = pd.read_csv(summary_path, encoding="utf-8-sig")
    summarized_track_ids = set(
        summaries.loc[
            summaries["emotion_summary"].fillna("").astype(str).str.strip().ne(""),
            "track_id",
        ].astype(str)
    )
    seed_has_lyrics = catalog["track_id"].astype(str).isin(summarized_track_ids)
    artist_names = catalog["artist"].fillna("").astype(str).str.strip()

    per_seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(catalog.itertuples(index=False), start=1):
        seed_song_id = int(seed.song_id)
        rankings = model.recommend_from_items_for_weights(
            [seed_song_id], list(WEIGHTS), top_k=TOP_K
        )
        baseline_ids = set(
            rankings[0.0]["track_id"].astype(str).tolist()
        )
        for weight in WEIGHTS:
            results = rankings[weight]
            recommended_rows = results["song_id"].to_numpy(dtype=np.int64)
            recommended_ids = results["track_id"].astype(str)
            recommendation_has_lyrics = recommended_ids.isin(
                summarized_track_ids
            )
            per_seed_rows.append(
                {
                    "seed_track_id": str(seed.track_id),
                    "seed_title": str(seed.title),
                    "seed_dataset_source": str(
                        getattr(seed, "dataset_source", "")
                    ),
                    "seed_genre": str(getattr(seed, "genre", "")),
                    "seed_has_approved_lyrics_summary": bool(
                        seed_has_lyrics.iloc[seed_index - 1]
                    ),
                    "emotion_weight": weight,
                    "baseline_top10_overlap": float(
                        recommended_ids.isin(baseline_ids).sum() / TOP_K
                    ),
                    "ild_at_10": intra_list_diversity_at_k(
                        recommended_rows, model.baseline.item_features, TOP_K
                    ),
                    "mean_embedding_similarity": float(
                        results["emotion_score"].mean()
                    ),
                    "unique_artists_at_10": int(
                        results["artist"].fillna("").astype(str).str.strip().replace("", np.nan).nunique()
                    ),
                    "recommendation_approved_lyrics_share": float(
                        recommendation_has_lyrics.mean()
                    ),
                }
            )
        if seed_index % 100 == 0 or seed_index == len(catalog):
            print(
                f"[FULL CATALOG] {seed_index}/{len(catalog)} seeds",
                flush=True,
            )

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed_path = output_dir / "weight_comparison_by_seed.csv"
    per_seed.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for weight, group in per_seed.groupby("emotion_weight", sort=True):
        lyric_seed_group = group.loc[
            group["seed_has_approved_lyrics_summary"]
        ]
        summary_rows.append(
            {
                "emotion_weight": float(weight),
                "n_seeds_all_catalog": int(group["seed_track_id"].nunique()),
                "n_seeds_with_approved_lyrics": int(
                    lyric_seed_group["seed_track_id"].nunique()
                ),
                "baseline_top10_overlap_mean": float(
                    group["baseline_top10_overlap"].mean()
                ),
                "ild_at_10_mean": float(group["ild_at_10"].mean()),
                "mean_embedding_similarity": float(
                    group["mean_embedding_similarity"].mean()
                ),
                "unique_artists_at_10_mean": float(
                    group["unique_artists_at_10"].mean()
                ),
                "recommendation_approved_lyrics_share_mean": float(
                    group["recommendation_approved_lyrics_share"].mean()
                ),
                "ild_at_10_mean_approved_lyrics_seeds": float(
                    lyric_seed_group["ild_at_10"].mean()
                ),
                "baseline_top10_overlap_mean_approved_lyrics_seeds": float(
                    lyric_seed_group["baseline_top10_overlap"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "weight_comparison.csv"
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": "full recommendation catalog",
        "catalog_track_count": int(len(catalog)),
        "approved_lyrics_summary_track_count": int(seed_has_lyrics.sum()),
        "metadata_only_track_count": int(len(catalog) - seed_has_lyrics.sum()),
        "emotion_weights": list(WEIGHTS),
        "top_k": TOP_K,
        "comparison_seed_count": int(len(catalog)),
        "summary_csv": summary_csv_path.name,
        "per_seed_csv": per_seed_path.name,
        "interpretation_warning": (
            "Only tracks with approved lyric summaries have a lyrics-derived "
            "emotion signal. Other tracks use metadata-only embeddings. Ranking "
            "metrics do not measure listener preference or recommendation accuracy."
        ),
    }
    manifest_path = output_dir / "full_catalog_experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Compared {len(WEIGHTS)} weights across {len(catalog)} catalog seeds; "
        f"{int(seed_has_lyrics.sum())} seeds have approved lyric summaries."
    )
    print(summary.to_string(index=False))
    print("Saved full-catalog results.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare emotion-aware weights on music recommendation tracks."
    )
    parser.add_argument(
        "--scope",
        choices=("dataViet", "all"),
        default="dataViet",
        help="Compare the approved dataViet sample or the full catalog.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_USER_HISTORY)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    if args.scope == "all":
        run_full_catalog_experiment(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir / "full_catalog",
        )
        return
    run_experiment(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        history_path=args.history,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
