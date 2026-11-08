"""Score blinded recommendations with reproducible local recommendation metrics."""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .data_generator import load_song_catalog
    from .evaluator import intra_list_diversity_at_k
    from .models.emotion_aware import EmotionAwareRecommender
else:
    from data_generator import load_song_catalog
    from evaluator import intra_list_diversity_at_k
    from models.emotion_aware import EmotionAwareRecommender


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "emotion_artifacts"
DEFAULT_EXPERIMENT_DIR = DEFAULT_ARTIFACT_DIR / "experiment"
SCORING_METHOD = "local_metrics_v1"
TOP_K = 10
SCORE_COLUMNS = (
    "mood_fit_1to5",
    "relevance_1to5",
    "variety_1to5",
)
OUTPUT_COLUMNS = (
    "seed_track_id",
    "list_code",
    *SCORE_COLUMNS,
    "emotion_summary_similarity_mean",
    "content_similarity_mean",
    "ild_at_10",
    "unique_artists",
    "unique_genres",
    "artist_metadata_coverage",
    "confidence",
    "rationale",
    "scoring_method",
    "input_sha256",
)


def _input_signature(
    seed: dict[str, str],
    seed_summary: str,
    list_code: str,
    recommendations: str,
    summaries: dict[str, str],
) -> str:
    tracks = []
    for track in json.loads(recommendations):
        track_id = str(track["track_id"])
        if track_id not in summaries:
            raise ValueError(f"No emotion summary found for recommended track {track_id!r}.")
        tracks.append(
            {"track": track, "emotion_summary": summaries[track_id]}
        )
    payload = json.dumps(
        [SCORING_METHOD, seed, seed_summary, list_code, tracks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(file, index=False)
    temporary.replace(path)


def _ordinal_scores(values: list[float]) -> list[int]:
    """Map metric ranks among one seed's blinded lists to ordinal scores 1-5."""
    if not values:
        return []
    if len(values) == 1 or max(values) == min(values):
        return [3] * len(values)
    ranks = pd.Series(values, dtype="float64").rank(method="average").to_numpy()
    denominator = len(values) - 1
    return [
        max(1, min(5, math.floor(1 + 4 * (rank - 1) / denominator + 0.5)))
        for rank in ranks
    ]


def _load_summary_vectors(
    artifact_dir: Path, track_ids: list[str]
) -> dict[str, np.ndarray]:
    manifest_path = artifact_dir / "manifest.json"
    ids_path = artifact_dir / "song_ids.json"
    if not manifest_path.is_file() or not ids_path.is_file():
        manifest_path = artifact_dir / "manifest.sample.json"
        ids_path = artifact_dir / "song_ids.sample.json"
    if not manifest_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            f"Emotion artifact manifest and IDs were not found in {artifact_dir}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors_name = manifest.get("summary_embeddings_file")
    if not isinstance(vectors_name, str) or not vectors_name:
        raise ValueError(
            "Emotion-summary vectors are missing. Rebuild the emotion artifacts "
            "with the updated emotion_pipeline.py."
        )
    vectors_path = artifact_dir / vectors_name
    if not vectors_path.is_file():
        raise FileNotFoundError(f"Emotion-summary vectors not found: {vectors_path}")
    artifact_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path, allow_pickle=False)
    if (
        not isinstance(artifact_ids, list)
        or not all(isinstance(track_id, str) for track_id in artifact_ids)
        or len(artifact_ids) != len(set(artifact_ids))
        or vectors.ndim != 2
        or vectors.shape[0] != len(artifact_ids)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Emotion-summary vectors and track IDs are misaligned.")
    if not set(track_ids).issubset(artifact_ids):
        raise ValueError("Emotion-summary vectors do not cover the recommender catalog.")
    vector_by_track = dict(zip(artifact_ids, vectors))
    return {track_id: vector_by_track[track_id] for track_id in track_ids}


def _score_seed_lists(
    model: EmotionAwareRecommender,
    *,
    seed_id: str,
    lists: list[dict[str, str]],
    summary_statuses: dict[str, str],
    summary_vectors: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    if len(lists) != 5:
        raise ValueError(
            f"Seed {seed_id!r} must have exactly five blinded lists for "
            "within-seed comparison."
        )
    seed_matches = model.songs.index[
        model.songs["track_id"].astype(str).eq(seed_id)
    ]
    if len(seed_matches) != 1:
        raise ValueError(f"Expected one catalog entry for seed {seed_id!r}.")
    seed_song_id = model.songs.iloc[int(seed_matches[0])]["song_id"]
    catalog_rankings = model.recommend_from_items(
        [seed_song_id], top_k=len(model.songs)
    )
    scores_by_track = catalog_rankings.assign(
        track_id=catalog_rankings["track_id"].astype(str)
    ).set_index("track_id")
    songs_by_track = model.songs.set_index(model.songs["track_id"].astype(str))
    k = min(TOP_K, len(model.songs) - 1)
    metrics: list[dict[str, object]] = []
    for row in lists:
        recommendations = json.loads(row["recommendations"])
        if len(recommendations) != TOP_K:
            raise ValueError(
                f"Seed {seed_id!r}, list {row['list_code']!r} must contain "
                f"exactly {TOP_K} recommendations."
            )
        track_ids = [str(track["track_id"]) for track in recommendations]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(
                f"Seed {seed_id!r}, list {row['list_code']!r} contains duplicate tracks."
            )
        missing = set(track_ids) - set(scores_by_track.index)
        if missing:
            raise ValueError(
                "Recommended tracks do not match the emotion artifact catalog: "
                f"{sorted(missing)[:5]!r}."
            )

        selected_scores = scores_by_track.loc[track_ids]
        selected_songs = songs_by_track.loc[track_ids]
        song_rows = np.asarray(
            [
                model._row_by_song_id[song_id]
                for song_id in selected_songs["song_id"]
            ],
            dtype=np.int64,
        )
        content_mean = float(selected_scores["content_score"].mean())
        seed_vector = summary_vectors[seed_id]
        seed_norm = float(np.linalg.norm(seed_vector))
        recommendation_vectors = np.vstack(
            [summary_vectors[track_id] for track_id in track_ids]
        )
        recommendation_norms = np.linalg.norm(recommendation_vectors, axis=1)
        valid_vectors = recommendation_norms > 0
        if seed_norm > 0 and valid_vectors.any():
            cosine_scores = (
                recommendation_vectors[valid_vectors] @ seed_vector
            ) / (recommendation_norms[valid_vectors] * seed_norm)
            emotion_summary_mean = float(
                np.clip(cosine_scores, -1.0, 1.0).mean()
            )
        else:
            emotion_summary_mean = 0.0
        ild = intra_list_diversity_at_k(song_rows, model.baseline.item_features, k)
        artists = selected_songs["artist"].fillna("").astype(str).str.strip()
        genres = selected_songs.get(
            "genre", pd.Series("", index=selected_songs.index)
        ).fillna("").astype(str).str.strip()
        known_artists = artists[artists.ne("")]
        unique_artists = int(known_artists.nunique())
        unique_genres = int(genres[genres.ne("")].nunique())
        artist_coverage = float(artists.ne("").mean())
        bad_summary_count = sum(
            summary_statuses.get(track_id, "missing") != "ok"
            for track_id in [seed_id, *track_ids]
        )
        metrics.append(
            {
                "seed_track_id": seed_id,
                "list_code": str(row["list_code"]),
                "emotion_summary_similarity_mean": emotion_summary_mean,
                "content_similarity_mean": content_mean,
                "ild_at_10": ild,
                "unique_artists": unique_artists,
                "unique_genres": unique_genres,
                "artist_metadata_coverage": artist_coverage,
                "confidence": (
                    "low"
                    if bad_summary_count or artist_coverage < 0.8
                    else "medium"
                ),
                "_bad_summary_count": bad_summary_count,
            }
        )

    for score_column, metric_column in (
        ("mood_fit_1to5", "emotion_summary_similarity_mean"),
        ("relevance_1to5", "content_similarity_mean"),
        ("variety_1to5", "ild_at_10"),
    ):
        scores = _ordinal_scores(
            [float(row[metric_column]) for row in metrics]
        )
        for row, score in zip(metrics, scores):
            row[score_column] = score

    for metric in metrics:
        metric["rationale"] = (
            "Điểm 1-5 là thứ hạng tương đối giữa 5 danh sách của cùng seed. "
            "Cosine BGE-M3 giữa các tóm tắt cảm xúc (không phải nhãn cảm xúc "
            "đã kiểm chứng)="
            f"{metric['emotion_summary_similarity_mean']:.3f}; "
            f"TF-IDF content similarity={metric['content_similarity_mean']:.3f}; "
            f"ILD@10={metric['ild_at_10']:.3f}; "
            f"{metric['unique_artists']} nghệ sĩ có tên, "
            f"{metric['unique_genres']} thể loại, "
            f"độ đầy đủ nghệ sĩ={metric['artist_metadata_coverage']:.0%}. "
            f"Có {metric['_bad_summary_count']} summary thiếu hoặc fallback."
        )
        metric["scoring_method"] = SCORING_METHOD
        del metric["_bad_summary_count"]
    return metrics


def run_judgement(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    experiment_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
    output_path: str | Path | None = None,
    max_seeds: int | None = None,
) -> pd.DataFrame:
    """Score each seed's five blinded lists with local model metrics."""
    if max_seeds is not None and max_seeds < 1:
        raise ValueError("max_seeds must be positive when provided.")

    artifact_dir = Path(artifact_dir)
    experiment_dir = Path(experiment_dir)
    lists_path = experiment_dir / "blind_review_lists.csv"
    summaries_path = artifact_dir / "emotion_summaries.csv"
    output_path = (
        Path(output_path)
        if output_path
        else experiment_dir / "local_metrics_scores.csv"
    )
    if not lists_path.is_file():
        raise FileNotFoundError(f"Blind recommendation file not found: {lists_path}")
    if not summaries_path.is_file():
        raise FileNotFoundError(f"Emotion summaries not found: {summaries_path}")

    lists_frame = pd.read_csv(lists_path, encoding="utf-8-sig").fillna("")
    required_list_columns = {
        "seed_track_id", "seed_title", "seed_artist", "seed_genre",
        "list_code", "recommendations",
    }
    if not required_list_columns.issubset(lists_frame.columns):
        raise ValueError(
            "Blind recommendation file is missing required columns: "
            f"{sorted(required_list_columns - set(lists_frame.columns))!r}."
        )
    summaries_frame = pd.read_csv(summaries_path, encoding="utf-8-sig").fillna("")
    if not {"track_id", "emotion_summary"}.issubset(summaries_frame.columns):
        raise ValueError("Emotion summary file needs track_id and emotion_summary.")
    summaries = dict(
        zip(
            summaries_frame["track_id"].astype(str),
            summaries_frame["emotion_summary"].astype(str),
        )
    )
    if "summary_status" in summaries_frame:
        summary_statuses = dict(
            zip(
                summaries_frame["track_id"].astype(str),
                summaries_frame["summary_status"].astype(str),
            )
        )
    else:
        summary_statuses = {track_id: "ok" for track_id in summaries}

    rows_by_seed: dict[str, list[dict[str, str]]] = {}
    for row in lists_frame.to_dict(orient="records"):
        seed_id = str(row["seed_track_id"])
        if seed_id not in summaries:
            raise ValueError(f"No emotion summary found for seed {seed_id!r}.")
        seed = {
            "title": str(row["seed_title"]),
            "artist": str(row["seed_artist"]),
            "genre": str(row["seed_genre"]),
        }
        row["input_sha256"] = _input_signature(
            seed,
            summaries[seed_id],
            str(row["list_code"]),
            str(row["recommendations"]),
            summaries,
        )
        rows_by_seed.setdefault(seed_id, []).append(row)
    for seed_id, seed_rows in rows_by_seed.items():
        codes = [row["list_code"] for row in seed_rows]
        if len(codes) != len(set(codes)):
            raise ValueError(f"Duplicate list_code found for seed {seed_id!r}.")
    if max_seeds is not None:
        selected_seed_ids = sorted(rows_by_seed)[:max_seeds]
    else:
        selected_seed_ids = sorted(rows_by_seed)

    current_by_key: dict[tuple[str, str], dict[str, object]] = {}
    if output_path.is_file():
        previous = pd.read_csv(output_path, encoding="utf-8-sig").fillna("")
        if set(OUTPUT_COLUMNS).issubset(previous.columns):
            for row in previous.to_dict(orient="records"):
                current_by_key[(str(row["seed_track_id"]), str(row["list_code"]))] = row

    expected_signatures = {
        (seed_id, row["list_code"]): row["input_sha256"]
        for seed_id, rows in rows_by_seed.items()
        for row in rows
    }
    current_by_key = {
        key: row
        for key, row in current_by_key.items()
        if (
            expected_signatures.get(key) == row.get("input_sha256")
            and row.get("scoring_method") == SCORING_METHOD
        )
    }
    catalog = load_song_catalog()
    recommender = EmotionAwareRecommender.from_artifacts(
        catalog, artifact_dir, emotion_weight=0.5
    )
    summary_vectors = _load_summary_vectors(
        artifact_dir, recommender.songs["track_id"].astype(str).tolist()
    )

    for index, seed_id in enumerate(selected_seed_ids, start=1):
        seed_rows = rows_by_seed[seed_id]
        if all(
            (seed_id, row["list_code"]) in current_by_key
            for row in seed_rows
        ):
            continue
        scored_lists = _score_seed_lists(
            recommender,
            seed_id=seed_id,
            lists=seed_rows,
            summary_statuses=summary_statuses,
            summary_vectors=summary_vectors,
        )
        for score in scored_lists:
            list_code = str(score["list_code"])
            score["input_sha256"] = next(
                row["input_sha256"]
                for row in seed_rows
                if row["list_code"] == list_code
            )
            current_by_key[(seed_id, list_code)] = score
        ordered_rows = [
            current_by_key[key]
            for key in sorted(current_by_key)
            if key in expected_signatures
        ]
        _atomic_write_csv(output_path, ordered_rows)
        print(
            f"[JUDGE] {index}/{len(selected_seed_ids)} seeds scored; "
            f"saved {output_path.name}",
            flush=True,
        )

    result_rows = [
        current_by_key[key]
        for key in sorted(current_by_key)
        if key in expected_signatures
    ]
    _atomic_write_csv(output_path, result_rows)
    return pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)


def summarize_by_weight(
    *,
    experiment_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
    scores_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Unblind completed scores and aggregate them by emotion weight."""
    experiment_dir = Path(experiment_dir)
    scores_path = (
        Path(scores_path)
        if scores_path else experiment_dir / "local_metrics_scores.csv"
    )
    output_path = (
        Path(output_path)
        if output_path
        else experiment_dir / "local_metrics_summary_by_weight.csv"
    )
    key_path = experiment_dir / "blind_review_key.csv"
    lists_path = experiment_dir / "blind_review_lists.csv"
    summaries_path = experiment_dir.parent / "emotion_summaries.csv"
    for path in (scores_path, key_path, lists_path, summaries_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required review file not found: {path}")

    scores = pd.read_csv(scores_path, encoding="utf-8-sig")
    key = pd.read_csv(key_path, encoding="utf-8-sig")
    lists = pd.read_csv(lists_path, encoding="utf-8-sig").fillna("")
    summaries_frame = pd.read_csv(summaries_path, encoding="utf-8-sig").fillna("")
    required = {"seed_track_id", "list_code"}
    if not required.issubset(scores.columns) or not required.issubset(key.columns):
        raise ValueError("Scores and blind key must contain seed_track_id and list_code.")
    if "input_sha256" not in scores:
        raise ValueError("Scores are missing the input signature; rerun local scoring.")
    summaries = dict(
        zip(
            summaries_frame["track_id"].astype(str),
            summaries_frame["emotion_summary"].astype(str),
        )
    )
    current_inputs: dict[tuple[str, str], str] = {}
    for row in lists.to_dict(orient="records"):
        seed_id = str(row["seed_track_id"])
        if seed_id not in summaries:
            raise ValueError(f"No emotion summary found for seed {seed_id!r}.")
        seed = {
            "title": str(row["seed_title"]),
            "artist": str(row["seed_artist"]),
            "genre": str(row["seed_genre"]),
        }
        current_inputs[(seed_id, str(row["list_code"]))] = _input_signature(
            seed,
            summaries[seed_id],
            str(row["list_code"]),
            str(row["recommendations"]),
            summaries,
        )
    score_inputs = {
        (str(row["seed_track_id"]), str(row["list_code"])): str(row["input_sha256"])
        for row in scores.to_dict(orient="records")
    }
    if score_inputs != current_inputs:
        raise ValueError(
            "Scores do not match the current blinded lists/summaries. "
            "Rerun local scoring before unblinding."
        )
    if len(scores) != len(key):
        raise ValueError(
            "Not all blinded lists have been scored yet; finish judging before "
            "opening the weight key."
        )
    joined = scores.merge(
        key,
        on=["seed_track_id", "list_code"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Scored lists do not exactly match the blind key; regenerate or "
            "complete scoring before unblinding."
        )
    summary = (
        joined.groupby(["emotion_weight", "system"], as_index=False)
        .agg(
            n_lists=("list_code", "count"),
            mood_fit_mean=("mood_fit_1to5", "mean"),
            relevance_mean=("relevance_1to5", "mean"),
            variety_mean=("variety_1to5", "mean"),
            low_confidence_share=(
                "confidence",
                lambda values: values.eq("low").mean(),
            ),
            emotion_summary_similarity_mean=(
                "emotion_summary_similarity_mean",
                "mean",
            ),
            content_similarity_mean=("content_similarity_mean", "mean"),
            ild_at_10_mean=("ild_at_10", "mean"),
            artist_metadata_coverage_mean=("artist_metadata_coverage", "mean"),
            unique_artists_mean=("unique_artists", "mean"),
            unique_genres_mean=("unique_genres", "mean"),
        )
        .sort_values("emotion_weight")
        .reset_index(drop=True)
    )
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score blinded recommendation lists with local model metrics."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="Score only the first N seed groups (useful for a small pilot).",
    )
    args = parser.parse_args()
    scores = run_judgement(
        artifact_dir=args.artifact_dir,
        experiment_dir=args.experiment_dir,
        output_path=args.output,
        max_seeds=args.max_seeds,
    )
    print(f"Saved {len(scores)} list scores.")


if __name__ == "__main__":
    main()
