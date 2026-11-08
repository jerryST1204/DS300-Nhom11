"""Run recommendations from the most recently trained model artifacts."""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __package__:
    from .model_bundle import TrainedModelBundle
else:
    from model_bundle import TrainedModelBundle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend tracks from the curated VMusic + dataViet catalog."
    )
    parser.add_argument(
        "--track-id",
        action="append",
        dest="track_ids",
        help="A liked track ID; repeat this option to provide multiple seeds.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--emotion-aware",
        action="store_true",
        help=(
            "Use the emotion-aware demo (emotion weight 0.5; limited to tracks "
            "in the generated emotion artifacts)."
        ),
    )
    args = parser.parse_args()
    if args.emotion_aware and not args.track_ids:
        parser.error("--emotion-aware requires at least one --track-id.")
    return args


def main() -> None:
    args = _arguments()
    models = TrainedModelBundle.from_directory()
    if args.emotion_aware:
        recommendations = models.recommend_emotion_aware(
            track_ids=args.track_ids,
            top_k=args.top_k,
            emotion_weight=0.5,
        )
        print(
            f"Emotion-aware recommendations (emotion weight=0.5) "
            f"based on {len(args.track_ids)} liked track(s):"
        )
    elif args.track_ids:
        recommendations = models.recommend_new_user(
            liked_track_ids=args.track_ids,
            top_k=args.top_k,
        )
        print(f"Recommendations based on {len(args.track_ids)} liked track(s):")
    else:
        recommendations = models.recommend_popular(top_k=args.top_k)
        print("Popular tracks across the catalog:")
        print("Pass --track-id TRACK_ID to personalize from liked tracks.")

    for rank, item in enumerate(recommendations, start=1):
        genre = f" [{item['genre']}]" if item.get("genre") else ""
        print(
            f"{rank:>2}. {item['title']} — {item['artist']}{genre} "
            f"(score={item['score']:.4f}, track_id={item['track_id']})"
        )


if __name__ == "__main__":
    main()
