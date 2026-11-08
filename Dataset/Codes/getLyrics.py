
import pandas as pd
import requests
import time
import re
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "clean_metadata_no_timestamp.csv"
OUTPUT_FILE = DATA_DIR / "clean_metadata_with_lyrics.csv"
OUTPUT_PARQUET_FILE = DATA_DIR / "clean_metadata_with_lyrics.parquet"
CHECKPOINT_EVERY = 10
MAX_RETRIES = 5

LRCLIB_URL = "https://lrclib.net/api/get"

HEADERS = {
    "User-Agent": "VietnameseMusicRecommendation/1.0 (student-project)"
}
processed_requests = 0


def save_checkpoint(dataframe):
    temporary_file = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.tmp{OUTPUT_FILE.suffix}"
    )
    dataframe.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig"
    )
    temporary_file.replace(OUTPUT_FILE)


# CLEAN TITLE FOR SEARCH

def prepare_title(title):

    if pd.isna(title):
        return ""

    title = str(title).strip()

    # bỏ một số phần dư còn sót lại
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
        title = re.sub(
            pattern,
            "",
            title,
            flags=re.IGNORECASE
        )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip(" |-_")

# CLEAN ARTIST FOR SEARCH

def prepare_artist(artist):

    if pd.isna(artist):
        return ""

    artist = str(artist).strip()

    artist = re.sub(
        r"\bofficial\b",
        "",
        artist,
        flags=re.IGNORECASE
    )

    artist = re.sub(
        r"\s+",
        " ",
        artist
    )

    return artist.strip(" |-_")

# REQUEST LRCLIB

def get_retry_delay(response, attempt):
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
                    (retry_at - datetime.now(timezone.utc)).total_seconds()
                )
            except (TypeError, ValueError, OverflowError):
                pass

    return min(2 ** (attempt - 1), 30)


def get_lyrics_from_lrclib(
    title,
    artist,
    duration
):

    if not title or not artist:
        return "not_found", None, None

    params = {
        "track_name": title,
        "artist_name": artist
    }

    try:
        duration = int(float(duration))

        if 1 <= duration <= 3600:
            params["duration"] = duration

    except (TypeError, ValueError):
        pass

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                LRCLIB_URL,
                params=params,
                headers=HEADERS,
                timeout=15
            )
        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                print(
                    f"\nNetwork error after {MAX_RETRIES} attempts: "
                    f"{title} - {artist}: {error}"
                )
                return "error", None, None

            wait_time = min(2 ** (attempt - 1), 30)
            print(
                f"\nNetwork error. Retry {attempt}/{MAX_RETRIES - 1} "
                f"after {wait_time}s: {title} - {artist}"
            )
            time.sleep(wait_time)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as error:
                print(
                    f"\nInvalid JSON response for {title} - {artist}: "
                    f"{error}"
                )
                return "error", None, None

            if not isinstance(data, dict):
                print(
                    f"\nUnexpected response format for {title} - {artist}"
                )
                return "error", None, None

            # CHỈ lấy lyrics không timestamp
            lyrics = data.get("plainLyrics")
            if not lyrics:
                return "not_found", None, None

            return "found", lyrics, data.get("id")

        if response.status_code == 404:
            return "not_found", None, None

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == MAX_RETRIES:
                print(
                    f"\nHTTP {response.status_code} after "
                    f"{MAX_RETRIES} attempts: {title} - {artist}"
                )
                return "error", None, None

            wait_time = get_retry_delay(response, attempt)
            print(
                f"\nHTTP {response.status_code}. Retry "
                f"{attempt}/{MAX_RETRIES - 1} after {wait_time:.1f}s: "
                f"{title} - {artist}"
            )
            time.sleep(wait_time)
            continue

        print(
            f"\nHTTP {response.status_code}: "
            f"{title} - {artist}"
        )
        return "error", None, None
    
# Load the input as the source of truth, then restore saved lyric progress.
df = pd.read_csv(INPUT_FILE)

if OUTPUT_FILE.exists():
    saved_df = pd.read_csv(OUTPUT_FILE)

    if len(saved_df) != len(df):
        raise ValueError(
            f"Không thể tiếp tục: {OUTPUT_FILE} có "
            f"{len(saved_df)} dòng, nhưng input có {len(df)} dòng."
        )

    key_columns = [
        column
        for column in ("id", "title", "artist")
        if column in df.columns and column in saved_df.columns
    ]

    if not key_columns:
        raise ValueError(
            "Không thể xác nhận file tiến trình khớp với input "
            "(thiếu cột id/title/artist)."
        )

    for column in key_columns:
        input_values = df[column].astype("string").fillna("")
        saved_values = saved_df[column].astype("string").fillna("")

        if not input_values.equals(saved_values):
            raise ValueError(
                f"Không thể tiếp tục: dữ liệu cột '{column}' trong "
                "file kết quả không khớp với input."
            )

    for column in ("lyrics", "lyrics_source", "lrclib_id"):
        if column in saved_df.columns:
            df[column] = saved_df[column]

    legacy_missing = df["lyrics_source"].eq("missing")
    if legacy_missing.any():
        df.loc[legacy_missing, "lyrics_source"] = None
        print(
            f"Đang thử lại {legacy_missing.sum()} bài từng được đánh dấu "
            "missing bởi phiên bản cũ."
        )

    print(f"Đã khôi phục tiến trình từ: {OUTPUT_FILE}")

print(
    "Total songs:",
    len(df)
)

#check có lyrics rồi thì không gọi LRCLIB
if "lyrics" not in df.columns:
    df["lyrics"] = None

if "lyrics_source" not in df.columns:
    df["lyrics_source"] = None

if "lrclib_id" not in df.columns:
    df["lrclib_id"] = None

# 1179 bài đã có lyrics, lyrics_source = subtitle 
for index, row in df.iterrows():

    subtitle = row.get(
        "subtitle_clean"
    )

    if pd.notna(subtitle):

        subtitle = str(
            subtitle
        ).strip()

        if subtitle:

            existing_lyrics = df.at[index, "lyrics"]
            if pd.notna(existing_lyrics) and str(existing_lyrics).strip():
                continue

            df.at[index, "lyrics"] = subtitle

            df.at[
                index,
                "lyrics_source"
            ] = "subtitle"

# gọi LRCLIB cho các bài thiếu lyrics
missing_count = (
    df["lyrics"]
    .isna()
    |
    df["lyrics"]
    .astype(str)
    .str.strip()
    .eq("")
).sum()

print(
    "Songs missing lyrics:",
    missing_count
)


for index, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    # ĐÃ CÓ LYRICS -> SKIP

    existing_lyrics = row.get(
        "lyrics"
    )

    if pd.notna(existing_lyrics):

        if str(existing_lyrics).strip():
            continue

    if row.get("lyrics_source") == "not_found":
        continue

    # PREPARE METADATA

    title = prepare_title(
        row.get(
            "title_clean",
            row.get("title")
        )
    )

    artist = prepare_artist(
        row.get(
            "artist_clean",
            row.get("artist")
        )
    )

    duration = row.get(
        "duration"
    )

    # LRCLIB

    status, lyrics, lrclib_id = (
        get_lyrics_from_lrclib(
            title,
            artist,
            duration
        )
    )
    # FOUND

    if status == "found":

        df.at[
            index,
            "lyrics"
        ] = lyrics

        df.at[
            index,
            "lrclib_id"
        ] = lrclib_id

        df.at[
            index,
            "lyrics_source"
        ] = "lrclib"

    elif status == "not_found":

        df.at[
            index,
            "lyrics_source"
        ] = "not_found"

    processed_requests += 1
    if processed_requests % CHECKPOINT_EVERY == 0:
        save_checkpoint(df)

    # LRCLIB khuyên delay batch
    time.sleep(0.35)

save_checkpoint(df)

temporary_parquet_file = OUTPUT_PARQUET_FILE.with_name(
    f"{OUTPUT_PARQUET_FILE.stem}.tmp{OUTPUT_PARQUET_FILE.suffix}"
)
df.to_parquet(temporary_parquet_file, index=False)
temporary_parquet_file.replace(OUTPUT_PARQUET_FILE)


print("\n======================")
print("LYRICS REPORT")
print("======================")


total = len(df)

subtitle_count = (
    df["lyrics_source"]
    .eq("subtitle")
    .sum()
)

lrclib_count = (
    df["lyrics_source"]
    .eq("lrclib")
    .sum()
)

missing_count = (
    df["lyrics_source"]
    .eq("not_found")
    .sum()
)

has_lyrics = (
    df["lyrics"]
    .notna()
    .sum()
)


print(
    "Total songs:",
    total
)

print(
    "Subtitle lyrics:",
    subtitle_count
)

print(
    "LRCLIB lyrics:",
    lrclib_count
)

print(
    "Total with lyrics:",
    has_lyrics
)

print(
    "Still missing:",
    missing_count
)

print(
    "Coverage:",
    round(
        has_lyrics / total * 100,
        2
    ),
    "%"
)