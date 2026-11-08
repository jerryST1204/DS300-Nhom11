"""Clean and quality-check song lyrics without overwriting the original lyrics.

Usage:
    python clean_lyrics.py
    python clean_lyrics.py --input data/VMusic_lyrics_checkpoint_V3_2.csv
    python clean_lyrics.py --input path/to/input.csv --output-dir path/to/output

The script keeps every input row and every original column. It adds cleaned text,
quality flags, review reasons, duplicate statistics, and a vector-use flag.
By default, it also cleans dataViet/tracks_with_lyrics.csv and writes an appended
all-sources dataset to data/lyrics_all_sources_cleaned.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


# Lines matching these patterns are treated as channel/creator promotion, not lyrics.
# Patterns are deliberately line-based to avoid deleting a legitimate lyric merely
# because it contains a word such as "follow", "subscribe", or "Instagram".
PROMO_LINE_PATTERNS = [
    re.compile(r"^\s*(?:hãy\s*)?(?:đăng\s*ký|dang\s*ky)\s*kênh\b.*$", re.I),
    re.compile(r"^.*(?:đăng\s*ký\s*kênh|đăng\s*ký\s+để\s+nhận\s+(?:video|bài\s*hát)|subscribe\s+(?:to\s+)?(?:my|our|the)\s+channel|subcr[i]?be|subcribe).*$", re.I),
    re.compile(r"^\s*chúc\s+bạn\s+nghe\s+nhạc\s+vui\s+vẻ.*$", re.I),
    re.compile(r"^.*(?:if\s+this\s+is\s+not\s+your\s+language|auto[- ]translate\s+in\s+the\s+setting).*$", re.I),
    re.compile(r"^\s*(?:subscribe|please\s+subscribe|don't\s+forget\s+to\s+subscribe)\b.*$", re.I),
    re.compile(r"^\s*(?:nhấn|ấn|bấm)\s*(?:vào\s*)?(?:nút\s*)?(?:đăng\s*ký|theo\s*dõi)\b.*$", re.I),
    re.compile(r"^\s*(?:đừng\s+quên\s+)?(?:nhấn|ấn|bấm)?\s*(?:like|thích)\b.*(?:share|chia\s*sẻ|đăng\s*ký|subscribe|theo\s*dõi).*?$", re.I),
    re.compile(r"^\s*(?:like\s*(?:,|and|&)\s*subscribe|like\s+and\s+share)\b.*$", re.I),
    re.compile(r"^\s*(?:follow\s+(?:me|us|my\s+channel|our\s+channel)|theo\s+dõi\s+(?:kênh|chúng\s+tôi|mình))\b.*$", re.I),
    re.compile(r"^\s*(?:ring\s+the\s+bell|nhấn\s+chuông|ấn\s+chuông)\b.*$", re.I),
    re.compile(r"^\s*(?:video\s+này\s+lên\s+án|khán\s+giả\s+vui\s+lòng\s+cân\s+nhắc|cảm\s+ơn\s+.*(?:theo\s+dõi|đăng\s+ký))\b.*$", re.I),
    re.compile(r"^\s*(?:đừng\s+quên\s+đăng\s*ký|don't\s+forget\s+to\s+like)\b.*$", re.I),
]

# Section headings and non-lyrical stage directions. Only whole lines are removed.
SECTION_LINE_RE = re.compile(
    r"^\s*[\[【(（]\s*(?:verse\s*\d*|chorus|pre[- ]?chorus|bridge|intro|outro|hook|refrain|interlude|instrumental|music|applause|音楽|伴奏)\s*[\]】)）]\s*$",
    re.I,
)

# Common timestamp forms: SRT/VTT cue lines, LRC tags, and standalone timestamps.
SRT_CUE_RE = re.compile(
    r"^\s*(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3}.*$"
)
STANDALONE_TIMESTAMP_RE = re.compile(
    r"^\s*(?:\[?\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]?|\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)\s*$"
)
LRC_PREFIX_RE = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
LEADING_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?\s*", re.M)
URL_RE = re.compile(r"(?i)https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")

ZERO_WIDTH_TRANSLATION = str.maketrans({
    "\ufeff": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2060": "",
    "\xa0": " ",
})


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def clean_lyrics(value: Any) -> tuple[str, dict[str, int]]:
    """Return cleaned lyrics and counts of removed artefacts."""
    counts = Counter({
        "promo_lines_removed": 0,
        "timestamp_lines_removed": 0,
        "url_lines_removed": 0,
        "section_lines_removed": 0,
        "empty_lines_removed": 0,
    })

    if is_missing(value):
        return "", dict(counts)

    text = unicodedata.normalize("NFC", str(value)).translate(ZERO_WIDTH_TRANSLATION)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_TAG_RE.sub(" ", text)

    # Some LRC files put timestamps directly before the lyric on the same line.
    text = LRC_PREFIX_RE.sub("", text)
    text = LEADING_TIMESTAMP_RE.sub("", text)

    output_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            counts["empty_lines_removed"] += 1
            continue

        if SRT_CUE_RE.match(line) or STANDALONE_TIMESTAMP_RE.match(line):
            counts["timestamp_lines_removed"] += 1
            continue

        if SECTION_LINE_RE.match(line):
            counts["section_lines_removed"] += 1
            continue

        if URL_RE.search(line):
            line = URL_RE.sub(" ", line).strip()
            counts["url_lines_removed"] += 1
            if not line:
                continue

        if any(pattern.match(line) for pattern in PROMO_LINE_PATTERNS):
            counts["promo_lines_removed"] += 1
            continue

        # Remove common lyric-video labels only when they appear as a prefix.
        line = re.sub(r"(?i)^\s*(?:official\s+)?lyrics?\s*[:：-]\s*", "", line).strip()
        if line:
            output_lines.append(line)

    # Collapse excessive blank lines while preserving verse separation.
    cleaned = "\n".join(output_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, dict(counts)


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    # Handles Vietnamese words separated by spaces and English contractions.
    return len(re.findall(r"\b[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?\b", text, flags=re.UNICODE))


def title_artist_only(text: str, title: Any, artist: Any) -> bool:
    normalized_text = re.sub(r"[^\wÀ-ỹ]+", " ", str(text).casefold()).strip()
    candidates = [title, artist]
    if not is_missing(title) and not is_missing(artist):
        candidates.extend([f"{title} {artist}", f"{artist} {title}"])
    normalized_candidates = {
        re.sub(r"[^\wÀ-ỹ]+", " ", str(x).casefold()).strip()
        for x in candidates if not is_missing(x)
    }
    return bool(normalized_text and normalized_text in normalized_candidates)


def build_quality_record(
    raw_lyrics: Any,
    cleaned: str,
    row: pd.Series,
    counts: dict[str, int],
    min_chars: int,
    min_words: int,
    match_threshold: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    raw_missing = is_missing(raw_lyrics)
    chars = len(cleaned)
    words = word_count(cleaned)

    if raw_missing:
        quality = "missing"
        reasons.append("no_original_lyrics")
    elif not cleaned:
        quality = "empty_after_cleaning"
        reasons.append("all_content_removed_or_empty")
    elif title_artist_only(cleaned, row.get("title"), row.get("artist")):
        quality = "review_title_only"
        reasons.append("content_is_only_title_or_artist")
    else:
        score = row.get("match_score")
        try:
            score_num = float(score) if not is_missing(score) else None
        except (TypeError, ValueError):
            score_num = None

        source_review = not is_missing(row.get("v2_review_flag"))
        review_needed = row.get("review_needed")
        if not source_review and not is_missing(review_needed):
            source_review = str(review_needed).strip().casefold() not in {
                "no", "false", "0", "n",
            }
        if source_review:
            reasons.append("source_marked_unverified_in_input")
        if score_num is not None and score_num < match_threshold:
            reasons.append("low_match_score")
        if chars < min_chars or words < min_words:
            reasons.append("short_lyrics_manual_review")
        if counts["promo_lines_removed"]:
            reasons.append("promotional_lines_removed")
        if counts["timestamp_lines_removed"]:
            reasons.append("timestamp_lines_removed")
        if counts["url_lines_removed"]:
            reasons.append("url_removed")

        if source_review:
            quality = "review_source_unverified"
        elif score_num is not None and score_num < match_threshold:
            quality = "review_low_match"
        elif chars < min_chars or words < min_words:
            quality = "review_short"
        elif counts["promo_lines_removed"] or counts["timestamp_lines_removed"] or counts["url_lines_removed"]:
            quality = "good_cleaned_noise_removed"
        else:
            quality = "good"

    # Keep all source values intact. This flag is only a recommended corpus filter.
    use_for_vector = (
        quality in {"good", "good_cleaned_noise_removed"}
        and chars >= min_chars
        and words >= min_words
    )

    return {
        "lyrics_clean_chars": chars,
        "lyrics_clean_words": words,
        "lyrics_quality": quality,
        "lyrics_review_reason": ";".join(reasons),
        "lyrics_use_for_vector": bool(use_for_vector),
    }


def clean_dataframe(
    df: pd.DataFrame,
    min_chars: int,
    min_words: int,
    match_threshold: float,
) -> pd.DataFrame:
    if "lyrics" not in df.columns:
        raise ValueError("CSV không có cột 'lyrics'.")

    df = df.copy()
    clean_results = [clean_lyrics(value) for value in df["lyrics"]]
    df["lyrics_clean"] = [cleaned for cleaned, _ in clean_results]
    clean_counts = [counts for _, counts in clean_results]
    count_df = pd.DataFrame(clean_counts, index=df.index).fillna(0).astype(int)
    for col in count_df.columns:
        df[col] = count_df[col]

    quality_records = [
        build_quality_record(
            raw_lyrics=row.get("lyrics"),
            cleaned=row["lyrics_clean"],
            row=row,
            counts=clean_counts[idx],
            min_chars=min_chars,
            min_words=min_words,
            match_threshold=match_threshold,
        )
        for idx, (_, row) in enumerate(df.iterrows())
    ]
    quality_df = pd.DataFrame(quality_records, index=df.index)
    for col in quality_df.columns:
        df[col] = quality_df[col]

    hashes = df["lyrics_clean"].map(normalized_hash)
    df["lyrics_clean_hash"] = hashes
    duplicate_counts = hashes.where(hashes.ne("")).map(hashes.value_counts()).fillna(0).astype(int)
    df["lyrics_clean_duplicate_count"] = duplicate_counts
    df["lyrics_clean_is_duplicate"] = hashes.ne("") & duplicate_counts.gt(1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and quality-check lyrics from both datasets.")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "data" / "VMusic_lyrics_checkpoint_V3_2.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--min-chars", type=int, default=150, help="Minimum cleaned character count for vector-use recommendation.")
    parser.add_argument("--min-words", type=int, default=25, help="Minimum cleaned word count for vector-use recommendation.")
    parser.add_argument("--match-threshold", type=float, default=0.80, help="Rows with a lower numeric match_score are flagged for review.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy input CSV: {args.input}")
    if args.min_chars < 0 or args.min_words < 0:
        raise ValueError("--min-chars và --min-words phải >= 0")
    if not 0 <= args.match_threshold <= 1:
        raise ValueError("--match-threshold phải nằm trong [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(__file__).resolve().parent
    data_viet_input = project_dir / "dataViet" / "tracks_with_lyrics.csv"
    if not data_viet_input.exists():
        raise FileNotFoundError(f"Không tìm thấy input CSV: {data_viet_input}")

    df = clean_dataframe(
        pd.read_csv(args.input, low_memory=False),
        args.min_chars,
        args.min_words,
        args.match_threshold,
    )
    df["dataset_source"] = "VMusic"
    data_viet_df = clean_dataframe(
        pd.read_csv(data_viet_input, low_memory=False),
        args.min_chars,
        args.min_words,
        args.match_threshold,
    )
    data_viet_df["dataset_source"] = "dataViet"

    input_stem = args.input.stem.replace("_checkpoint", "")
    output_csv = args.output_dir / f"{input_stem}_cleaned.csv"
    review_csv = args.output_dir / f"{input_stem}_lyrics_review.csv"
    vector_csv = args.output_dir / f"{input_stem}_vector_ready.csv"
    report_json = args.output_dir / f"{input_stem}_cleaning_report.json"
    report_txt = args.output_dir / f"{input_stem}_cleaning_report.txt"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    data_viet_output = data_viet_input.with_name("tracks_with_lyrics_cleaned.csv")
    data_viet_review_output = data_viet_input.with_name("tracks_with_lyrics_lyrics_review.csv")
    data_viet_vector_output = data_viet_input.with_name("tracks_with_lyrics_vector_ready.csv")
    data_viet_df.to_csv(data_viet_output, index=False, encoding="utf-8-sig")

    combined_df = pd.concat([df, data_viet_df], ignore_index=True, sort=False)
    combined_hashes = combined_df["lyrics_clean"].map(normalized_hash)
    combined_df["lyrics_clean_hash"] = combined_hashes
    combined_duplicate_counts = (
        combined_hashes.where(combined_hashes.ne(""))
        .map(combined_hashes.value_counts())
        .fillna(0)
        .astype(int)
    )
    combined_df["lyrics_clean_duplicate_count"] = combined_duplicate_counts
    combined_df["lyrics_clean_is_duplicate"] = (
        combined_hashes.ne("") & combined_duplicate_counts.gt(1)
    )
    combined_output = args.output_dir / "lyrics_all_sources_cleaned.csv"
    combined_df.to_csv(combined_output, index=False, encoding="utf-8-sig")

    # Review includes lyrics and matching metadata so a human can inspect uncertain rows.
    review_mask = (
        df["lyrics"].notna()
        & df["lyrics"].astype(str).str.strip().ne("")
        & ~df["lyrics_use_for_vector"]
    )
    preferred_review_cols = [
        c for c in [
            "dataset_source", "id", "track_id", "genre", "title", "artist",
            "lyrics_source", "lyrics_status", "lyrics_error", "matched_title",
            "matched_artist", "match_score", "v2_review_flag", "review_needed",
            "review_reason",
            "lyrics_quality", "lyrics_review_reason", "lyrics_clean_chars",
            "lyrics_clean_words", "lyrics", "lyrics_clean",
        ] if c in df.columns
    ]
    df.loc[review_mask, preferred_review_cols].to_csv(review_csv, index=False, encoding="utf-8-sig")
    df.loc[df["lyrics_use_for_vector"]].to_csv(vector_csv, index=False, encoding="utf-8-sig")

    data_viet_review_mask = (
        data_viet_df["lyrics"].notna()
        & data_viet_df["lyrics"].astype(str).str.strip().ne("")
        & ~data_viet_df["lyrics_use_for_vector"]
    )
    data_viet_review_cols = [c for c in preferred_review_cols if c in data_viet_df.columns]
    data_viet_df.loc[data_viet_review_mask, data_viet_review_cols].to_csv(
        data_viet_review_output, index=False, encoding="utf-8-sig"
    )
    data_viet_df.loc[data_viet_df["lyrics_use_for_vector"]].to_csv(
        data_viet_vector_output, index=False, encoding="utf-8-sig"
    )

    quality_counts = df["lyrics_quality"].value_counts(dropna=False).to_dict()
    source_counts = (df["lyrics_source"].fillna("<NA>").value_counts().to_dict()
                     if "lyrics_source" in df.columns else {})
    retrieval_status_counts = (df["lyrics_status"].fillna("<NA>").value_counts().to_dict()
                               if "lyrics_status" in df.columns else {})
    lyrics_error_counts = (df["lyrics_error"].fillna("<NA>").value_counts().to_dict()
                           if "lyrics_error" in df.columns else {})
    total = len(df)
    raw_nonempty = int(df["lyrics"].fillna("").astype(str).str.strip().ne("").sum())
    clean_nonempty = int(df["lyrics_clean"].str.strip().ne("").sum())
    vector_usable = int(df["lyrics_use_for_vector"].sum())
    duplicate_rows = int(df["lyrics_clean_is_duplicate"].sum())
    source_unverified = int(df["lyrics_quality"].eq("review_source_unverified").sum())
    low_match = int(df["lyrics_quality"].eq("review_low_match").sum())

    report = {
        "input_file": str(args.input),
        "output_file": str(output_csv),
        "data_viet_input_file": str(data_viet_input),
        "data_viet_output_file": str(data_viet_output),
        "combined_output_file": str(combined_output),
        "combined_total_songs": len(combined_df),
        "combined_source_counts": {
            str(k): int(v) for k, v in combined_df["dataset_source"].value_counts().items()
        },
        "combined_quality_counts": {
            str(k): int(v) for k, v in combined_df["lyrics_quality"].value_counts().items()
        },
        "review_file": str(review_csv),
        "vector_ready_file": str(vector_csv),
        "total_songs": total,
        "rows_with_original_lyrics": raw_nonempty,
        "rows_with_clean_lyrics": clean_nonempty,
        "rows_recommended_for_vectorization": vector_usable,
        "rows_for_manual_review_with_lyrics": int(review_mask.sum()),
        "rows_with_duplicate_clean_lyrics": duplicate_rows,
        "source_unverified_rows": source_unverified,
        "low_match_rows": low_match,
        "thresholds": {
            "min_chars": args.min_chars,
            "min_words": args.min_words,
            "match_score": args.match_threshold,
        },
        "quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "retrieval_status_counts": {str(k): int(v) for k, v in retrieval_status_counts.items()},
        "lyrics_error_counts": {str(k): int(v) for k, v in lyrics_error_counts.items()},
        "removed_content_totals": {
            col: int(df[col].sum()) for col in [
                "promo_lines_removed", "timestamp_lines_removed", "url_lines_removed",
                "section_lines_removed", "empty_lines_removed",
            ]
        },
        "notes": [
            "Original lyrics column is preserved unchanged; cleaned text is stored in lyrics_clean.",
            "No song rows are deleted.",
            "Duplicate lyrics are flagged, not removed, because covers/remixes may legitimately share lyrics.",
            "lyrics_use_for_vector is a conservative recommendation, not a claim that reviewed rows are invalid.",
            "api_error and provider_paused retrieval states are preserved from the input dataset.",
        ],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "VMusic LYRICS CLEANING REPORT",
        "=" * 34,
        f"Total songs: {total}",
        f"Rows with original lyrics: {raw_nonempty}",
        f"Rows with cleaned lyrics: {clean_nonempty}",
        f"Recommended for vectorization: {vector_usable}",
        f"Rows with lyrics needing manual review: {int(review_mask.sum())}",
        f"Rows flagged as duplicate lyrics: {duplicate_rows}",
        "",
        "Quality distribution:",
    ]
    lines += [f"  {k}: {v}" for k, v in sorted(quality_counts.items())]
    if retrieval_status_counts:
        lines += ["", "Original retrieval statuses:"]
        lines += [f"  {k}: {v}" for k, v in sorted(retrieval_status_counts.items())]
    if lyrics_error_counts:
        lines += ["", "Original retrieval errors:"]
        lines += [f"  {k}: {v}" for k, v in sorted(lyrics_error_counts.items())]
    lines += ["", "Removed content totals:"]
    lines += [f"  {col}: {int(df[col].sum())}" for col in [
        "promo_lines_removed", "timestamp_lines_removed", "url_lines_removed",
        "section_lines_removed", "empty_lines_removed",
    ]]
    lines += [
        "",
        "Important: lyrics is the original text; lyrics_clean is the cleaned text.",
        "Use lyrics_use_for_vector == True for the initial high-confidence lyrics corpus.",
        "Inspect the review CSV before deciding whether to include flagged rows.",
    ]
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nOutput files:")
    print(f"  Cleaned dataset: {output_csv}")
    print(f"  Manual review:   {review_csv}")
    print(f"  Vector-ready:    {vector_csv}")
    print(f"  dataViet cleaned: {data_viet_output}")
    print(f"  Combined dataset: {combined_output}")
    print(f"  dataViet review:  {data_viet_review_output}")
    print(f"  dataViet vector:  {data_viet_vector_output}")
    print(f"  JSON report:     {report_json}")
    print(f"  Text report:     {report_txt}")


if __name__ == "__main__":
    main()
