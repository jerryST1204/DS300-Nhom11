"""Fetch missing lyrics in the combined dataset using LRCLIB's plain-lyrics API.

The source dataset is never modified. Progress is saved to a checkpoint CSV so
the script can resume after interruption.

Usage:
    python get_missing_lyrics.py
    python get_missing_lyrics.py --input data/lyrics_all_sources_cleaned.csv
    python get_missing_lyrics.py --retry-not-found
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = None

from clean_lyrics import clean_dataframe, is_missing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "lyrics_all_sources_cleaned.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "lyrics_all_sources_with_missing_filled.csv"
LRCLIB_URL = "https://lrclib.net/api/get"
HEADERS = {
    "User-Agent": "VietnameseMusicRecommendation/1.0 (student-project)"
}
MAX_RETRIES = 5
MAX_CONSECUTIVE_ERRORS = 5
CHECKPOINT_EVERY = 10
REQUEST_DELAY_SECONDS = 0.35


class _ConsoleProgress:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.postfix: dict[str, Any] = {}

    def set_description_str(self, desc: str) -> None:
        self.desc = desc

    def set_postfix(self, **values: Any) -> None:
        self.postfix = values
        self._render()

    def update(self, amount: int = 1) -> None:
        self.current = min(self.current + amount, self.total)

    def _render(self) -> None:
        width = 24
        ratio = self.current / self.total if self.total else 1.0
        filled = int(width * ratio)
        bar = "=" * filled + "." * (width - filled)
        details = " ".join(
            f"{key}={value}" for key, value in self.postfix.items()
        )
        line = (
            f"\r{self.desc[:40]:<40} [{bar}] "
            f"{self.current}/{self.total} {self.unit} {details}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def close(self) -> None:
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()


def create_progress(total: int) -> Any:
    if _tqdm is not None:
        return _tqdm(
            total=total,
            desc="Searching lyrics",
            unit="tracks",
            dynamic_ncols=True,
            ascii=True,
        )
    return _ConsoleProgress(total, "Searching lyrics", "tracks")


def prepare_title(value: Any) -> str:
    if is_missing(value):
        return ""
    title = str(value).strip()
    patterns = [
        r"\(official.*?\)",
        r"\[official.*?\]",
        r"\(lyrics?.*?\)",
        r"\[lyrics?.*?\]",
        r"\(audio.*?\)",
        r"\[audio.*?\]",
        r"\(visualizer.*?\)",
        r"\[visualizer.*?\]",
    ]
    for pattern in patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip(" |-_")


def prepare_artist(value: Any) -> str:
    if is_missing(value):
        return ""
    artist = re.sub(r"\bofficial\b", "", str(value).strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", artist).strip(" |-_")


def get_retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(
                    0,
                    (retry_at - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return min(2 ** (attempt - 1), 30)


def get_lyrics_from_lrclib(
    title: str,
    artist: str,
    duration: Any,
) -> tuple[str, str | None, Any, str | None]:
    if not title or not artist:
        return "skipped_missing_metadata", None, None, None

    params: dict[str, Any] = {
        "track_name": title,
        "artist_name": artist,
    }
    if not is_missing(duration):
        try:
            duration_seconds = int(float(duration))
            if 1 <= duration_seconds <= 3600:
                params["duration"] = duration_seconds
        except (TypeError, ValueError, OverflowError):
            pass

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                LRCLIB_URL,
                params=params,
                headers=HEADERS,
                timeout=15,
            )
        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                return "error", None, None, str(error)
            time.sleep(min(2 ** (attempt - 1), 30))
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as error:
                return "error", None, None, f"Invalid JSON response: {error}"
            if not isinstance(data, dict):
                return "error", None, None, "Unexpected LRCLIB response format"
            lyrics = data.get("plainLyrics")
            if not isinstance(lyrics, str) or not lyrics.strip():
                return "not_found", None, None, None
            return "found", lyrics, data.get("id"), None

        if response.status_code == 404:
            return "not_found", None, None, None

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == MAX_RETRIES:
                return (
                    "error",
                    None,
                    None,
                    f"HTTP {response.status_code} after {MAX_RETRIES} attempts",
                )
            time.sleep(get_retry_delay(response, attempt))
            continue

        return "error", None, None, f"HTTP {response.status_code}"

    return "error", None, None, "LRCLIB request retries exhausted"


def first_non_missing(row: pd.Series, columns: tuple[str, ...]) -> Any:
    for column in columns:
        value = row.get(column)
        if not is_missing(value):
            return value
    return None


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)


def load_or_create_working_data(
    input_path: Path,
    checkpoint_path: Path,
) -> pd.DataFrame:
    source_df = pd.read_csv(input_path, low_memory=False)
    if "lyrics" not in source_df.columns:
        raise ValueError(f"Input CSV has no 'lyrics' column: {input_path}")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_df["lyrics_lookup_input_sha256"] = input_hash

    if checkpoint_path.exists():
        saved_df = pd.read_csv(checkpoint_path, low_memory=False)
        if len(saved_df) != len(source_df):
            raise ValueError(
                f"Checkpoint has {len(saved_df)} rows, but input has "
                f"{len(source_df)} rows: {checkpoint_path}"
            )
        if "lyrics_lookup_input_sha256" not in saved_df.columns:
            raise ValueError(
                f"Checkpoint has no input fingerprint and cannot be verified: "
                f"{checkpoint_path}"
            )
        saved_hashes = set(
            saved_df["lyrics_lookup_input_sha256"].dropna().astype(str)
        )
        if saved_hashes != {input_hash}:
            raise ValueError(
                "Checkpoint was created from a different or changed input; "
                "rename or move the old checkpoint before running."
            )

        identity_columns = [
            column
            for column in ("dataset_source", "id", "track_id", "title", "artist")
            if column in source_df.columns and column in saved_df.columns
        ]
        if not identity_columns:
            raise ValueError("Cannot verify checkpoint against the input file.")
        for column in identity_columns:
            source_values = source_df[column].astype("string").fillna("")
            saved_values = saved_df[column].astype("string").fillna("")
            if not source_values.equals(saved_values):
                raise ValueError(
                    f"Checkpoint does not match the input in column '{column}'; "
                    "rename or move the old checkpoint before running."
                )
        print(f"Resuming from checkpoint: {checkpoint_path}")
        return saved_df

    return source_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch missing lyrics for the combined VMusic/dataViet dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--retry-not-found",
        action="store_true",
        help="Retry rows previously returned as not found by LRCLIB.",
    )
    parser.add_argument(
        "--auto-retry-rounds",
        type=int,
        default=5,
        help="Automatic continuation rounds after API errors (default: 5).",
    )
    parser.add_argument(
        "--retry-wait-minutes",
        type=float,
        default=3,
        help="Minutes to wait between automatic continuation rounds (default: 3).",
    )
    parser.add_argument("--min-chars", type=int, default=150)
    parser.add_argument("--min-words", type=int, default=25)
    parser.add_argument("--match-threshold", type=float, default=0.80)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different.")
    if args.min_chars < 0 or args.min_words < 0:
        raise ValueError("--min-chars and --min-words must be >= 0")
    if not 0 <= args.match_threshold <= 1:
        raise ValueError("--match-threshold must be in [0, 1]")
    if args.auto_retry_rounds < 0:
        raise ValueError("--auto-retry-rounds must be >= 0")
    if args.retry_wait_minutes < 0:
        raise ValueError("--retry-wait-minutes must be >= 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_name(
        f"{args.output.stem}_checkpoint{args.output.suffix}"
    )
    df = load_or_create_working_data(args.input, checkpoint_path)

    for column in ("lyrics_source", "lrclib_id", "lyrics_status"):
        if column not in df.columns:
            df[column] = pd.NA
    for column in ("lyrics_lookup_status", "lyrics_lookup_error"):
        if column not in df.columns:
            df[column] = pd.NA
    for column in ("lyrics", "lyrics_source", "lrclib_id", "lyrics_status"):
        df[column] = df[column].astype("object")

    def get_pending_indices() -> list[Any]:
        missing = df["lyrics"].isna() | df["lyrics"].astype(str).str.strip().eq("")
        prior = df["lyrics_lookup_status"].fillna("").astype(str)
        instrumental_mask = (
            df["lyrics_status"].fillna("").astype(str).str.casefold().eq("instrumental")
        )
        df.loc[instrumental_mask & missing, "lyrics_lookup_status"] = (
            "skipped_instrumental"
        )
        eligible_mask = (
            missing
            & ~instrumental_mask
            & ~prior.eq("skipped_missing_metadata")
        )
        if not args.retry_not_found:
            eligible_mask &= ~prior.eq("not_found")
        return df.index[eligible_mask].tolist()

    missing_lyrics = df["lyrics"].isna() | df["lyrics"].astype(str).str.strip().eq("")
    initial_prior_status = df["lyrics_lookup_status"].fillna("").astype(str)
    initial_instrumental = (
        df["lyrics_status"].fillna("").astype(str).str.casefold().eq("instrumental")
    )
    skipped_indices = set(df.index[~missing_lyrics | initial_instrumental])
    skipped_indices.update(
        df.index[missing_lyrics & initial_prior_status.eq("skipped_missing_metadata")]
    )
    if not args.retry_not_found:
        skipped_indices.update(
            df.index[missing_lyrics & initial_prior_status.eq("not_found")]
        )

    pending_indices = get_pending_indices()
    print(f"Total tracks: {len(df)}")
    print(f"Missing lyrics to look up: {len(pending_indices)}")
    print(f"Skipped before lookup: {len(skipped_indices)}")

    attempts_since_checkpoint = 0
    attempted_count = 0
    found_count = 0
    error_count = 0
    auto_retry_rounds_used = 0
    stopped_after_api_errors = False
    rounds_total = args.auto_retry_rounds + 1
    for round_index in range(rounds_total):
        pending_indices = get_pending_indices()
        if not pending_indices:
            break

        consecutive_errors = 0
        stopped_after_api_errors = False
        progress = create_progress(len(pending_indices))
        for index in pending_indices:
            row = df.loc[index]
            title = prepare_title(first_non_missing(
                row, ("title_clean", "title", "title_query")
            ))
            artist = prepare_artist(first_non_missing(
                row, ("artist_clean", "artist")
            ))
            duration = first_non_missing(row, ("duration", "duration_seconds"))

            if not title or not artist:
                df.at[index, "lyrics_lookup_status"] = "skipped_missing_metadata"
                df.at[index, "lyrics_lookup_error"] = pd.NA
                skipped_indices.add(index)
                progress.update(1)
                progress.set_postfix(
                    searching=0,
                    searched=attempted_count,
                    skipped=len(skipped_indices),
                    found=found_count,
                    errors=error_count,
                )
                continue

            progress.set_postfix(
                searching=1,
                searched=attempted_count,
                skipped=len(skipped_indices),
                found=found_count,
                errors=error_count,
            )
            status, lyrics, lrclib_id, error = get_lyrics_from_lrclib(
                title, artist, duration
            )
            attempted_count += 1
            df.at[index, "lyrics_lookup_status"] = status
            df.at[index, "lyrics_lookup_error"] = error

            if status == "found":
                df.at[index, "lyrics"] = lyrics
                df.at[index, "lrclib_id"] = lrclib_id
                df.at[index, "lyrics_source"] = "lrclib"
                df.at[index, "lyrics_status"] = "found"
                consecutive_errors = 0
                found_count += 1
            elif status == "not_found":
                df.at[index, "lyrics_status"] = "missing"
                consecutive_errors = 0
                skipped_indices.add(index)
            elif status == "error":
                df.at[index, "lyrics_status"] = "api_error"
                consecutive_errors += 1
                error_count += 1
            else:
                consecutive_errors = 0
                skipped_indices.add(index)

            attempts_since_checkpoint += 1
            progress.update(1)
            progress.set_postfix(
                searching=0,
                searched=attempted_count,
                skipped=len(skipped_indices),
                found=found_count,
                errors=error_count,
            )
            if attempts_since_checkpoint >= CHECKPOINT_EVERY:
                save_checkpoint(df, checkpoint_path)
                attempts_since_checkpoint = 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                save_checkpoint(df, checkpoint_path)
                stopped_after_api_errors = True
                break
            time.sleep(REQUEST_DELAY_SECONDS)
        progress.close()

        save_checkpoint(df, checkpoint_path)
        pending_after_round = get_pending_indices()
        if not pending_after_round:
            stopped_after_api_errors = False
            break
        if round_index >= args.auto_retry_rounds:
            stopped_after_api_errors = True
            break

        auto_retry_rounds_used += 1
        wait_seconds = args.retry_wait_minutes * 60
        print(
            f"API errors remain for {len(pending_after_round)} tracks. "
            f"Automatically retrying in {args.retry_wait_minutes:g} minutes "
            f"(retry round {auto_retry_rounds_used}/{args.auto_retry_rounds})."
        )
        time.sleep(wait_seconds)

    save_checkpoint(df, checkpoint_path)

    cleaned_df = clean_dataframe(
        df,
        args.min_chars,
        args.min_words,
        args.match_threshold,
    )
    save_checkpoint(cleaned_df, checkpoint_path)

    temporary_output = args.output.with_name(
        f"{args.output.stem}.tmp{args.output.suffix}"
    )
    cleaned_df.to_csv(temporary_output, index=False, encoding="utf-8-sig")
    temporary_output.replace(args.output)
    remaining_retryable = len(get_pending_indices())

    report_path = args.output.with_name(f"{args.output.stem}_report.json")
    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "checkpoint_file": str(checkpoint_path),
        "total_rows": len(cleaned_df),
        "lookup_attempted": attempted_count,
        "lookup_found_this_run": found_count,
        "skipped_this_run": len(skipped_indices),
        "api_errors_this_run": error_count,
        "automatic_retry_rounds_used": auto_retry_rounds_used,
        "automatic_retry_rounds_limit": args.auto_retry_rounds,
        "retry_wait_minutes": args.retry_wait_minutes,
        "stopped_after_consecutive_api_errors": stopped_after_api_errors,
        "remaining_retryable_rows": remaining_retryable,
        "unprocessed_rows_this_run": remaining_retryable,
        "rows_with_lyrics": int(
            cleaned_df["lyrics"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "rows_still_missing": int(
            cleaned_df["lyrics"].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "lookup_status_counts": {
            str(key): int(value)
            for key, value in cleaned_df["lyrics_lookup_status"]
            .fillna("<NA>")
            .value_counts()
            .items()
        },
        "quality_counts": {
            str(key): int(value)
            for key, value in cleaned_df["lyrics_quality"]
            .fillna("<NA>")
            .value_counts()
            .items()
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCompleted.")
    print(f"Output file: {args.output}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Report: {report_path}")
    print(f"Lyrics still missing: {report['rows_still_missing']}")
    if remaining_retryable:
        print(
            "Automatic retry limit reached; rerun later to resume the "
            f"{remaining_retryable} tracks still affected by API errors."
        )


if __name__ == "__main__":
    main()
