"""Explore the song, genre, artist, and interaction data available in Music/."""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


SEGMENT_PATTERN = re.compile(
    r"(?P<track>.+?)[._-]segment[_-]?(?P<segment>\d+)\.mp3$",
    re.IGNORECASE,
)


def explore_music(music_dir: Path) -> tuple[dict[str, tuple[int, int]], int]:
    """Return per-genre (song, segment) counts and the total number of segments."""
    if not music_dir.is_dir():
        raise FileNotFoundError(f"Music directory not found: {music_dir}")

    genre_counts: dict[str, tuple[int, int]] = {}
    total_segments = 0

    for genre_dir in sorted(path for path in music_dir.iterdir() if path.is_dir()):
        songs: set[str] = set()
        segments = 0

        for audio_file in genre_dir.glob("*.mp3"):
            match = SEGMENT_PATTERN.fullmatch(audio_file.name)
            if match is None:
                raise ValueError(
                    f"Unexpected MP3 filename (expected a segment suffix): "
                    f"{audio_file}"
                )
            songs.add(match.group("track"))
            segments += 1

        if segments:
            genre_counts[genre_dir.name] = (len(songs), segments)
            total_segments += segments

    if not genre_counts:
        raise ValueError(f"No MP3 segments found in {music_dir}")

    return genre_counts, total_segments


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Summarize songs, genres, artists, and interactions in Music/."
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Music",
        help="Directory containing one subdirectory per genre (default: ./Music)",
    )
    args = parser.parse_args()

    try:
        genre_counts, total_segments = explore_music(args.music_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    total_songs = sum(song_count for song_count, _ in genre_counts.values())
    print("Khám phá dữ liệu âm nhạc")
    print(f"Số bài hát: {total_songs}")
    print(f"Số thể loại: {len(genre_counts)}")
    print("Số nghệ sĩ: Chưa xác định (không có metadata nghệ sĩ đáng tin cậy)")
    print("Lượt tương tác: Không có dữ liệu")
    print(f"Tổng số đoạn âm thanh: {total_segments}")
    print()
    print(f"{'Thể loại':<16} {'Bài hát':>8} {'Đoạn âm thanh':>16}")
    for genre, (song_count, segment_count) in genre_counts.items():
        print(f"{genre:<16} {song_count:>8} {segment_count:>16}")


if __name__ == "__main__":
    main()