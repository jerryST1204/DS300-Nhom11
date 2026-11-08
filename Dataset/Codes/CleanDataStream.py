import re
import unicodedata
from huggingface_hub import snapshot_download
import pandas as pd
import glob
from pathlib import Path


DATASET_NAME = "Toan-Minh-Duong-Son/vietnamese-music-dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Downloading metadata only...")

folder = snapshot_download(
    repo_id=DATASET_NAME,
    repo_type="dataset",

    # CHỈ tải thư mục data
    allow_patterns="data/*"
)

print("Metadata folder:")
print(folder)


files = sorted(
    glob.glob(folder + "/data/*.parquet")
)

print("Number of parquet files:", len(files))


# Đọc toàn bộ metadata
dfs = []

for file in files:

    print("Reading:", file)

    df = pd.read_parquet(file)

    dfs.append(df)


metadata = pd.concat(
    dfs,
    ignore_index=True
)

# Dataset có duplicate giữa các collection session
metadata = metadata.drop_duplicates(
    subset="id",
    keep="first"
)

#clean text
def clean_text(text):

    if pd.isna(text):
        return None

    text = str(text).strip()

    if not text:
        return None

    # Chuẩn hóa unicode tiếng Việt
    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Nhiều khoảng trắng -> một
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text

#clean title 
def clean_title(title):

    title = clean_text(title)

    if title is None:
        return None

    patterns = [
        r"\bofficial\s+music\s+video\b",
        r"\bofficial\s+video\b",
        r"\bofficial\s+mv\b",
        r"\bofficial\s+audio\b",
        r"\blyric\s+video\b",
        r"\blyrics\s+video\b",
        r"\bvisualizer\b",
        r"\blyrics?\b",
        r"\b4k\b",
        r"\bhd\b"
    ]

    for pattern in patterns:

        title = re.sub(
            pattern,
            "",
            title,
            flags=re.IGNORECASE
        )

    # bỏ ngoặc trống sau khi xóa
    title = re.sub(
        r"\(\s*\)",
        "",
        title
    )

    title = re.sub(
        r"\[\s*\]",
        "",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    title = title.strip(
        " |-_:;,."
    )

    if not title:
        return None

    return title
#clean artist 
def clean_artist(artist):

    artist = clean_text(artist)

    if artist is None:
        return None

    artist = re.sub(
        r"\bofficial\s+channel\b",
        "",
        artist,
        flags=re.IGNORECASE
    )

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

    artist = artist.strip(
        " |-_:;,."
    )

    if not artist:
        return None

    return artist

metadata["title_clean"] = (
    metadata["title"]
    .apply(clean_title)
)

metadata["artist_clean"] = (
    metadata["artist"]
    .apply(clean_artist)
)

metadata["duration"] = pd.to_numeric(
    metadata["duration"],
    errors="coerce"
)

MIN_DURATION = 30

MAX_DURATION = 15 * 60

metadata["duration_valid"] = (
    metadata["duration"].notna()
    &
    (metadata["duration"] >= MIN_DURATION)
    &
    (metadata["duration"] <= MAX_DURATION)
)

def clean_subtitle(subtitle):

    if pd.isna(subtitle):
        return None

    subtitle = str(subtitle).strip()

    if subtitle.lower() in [
        "",
        "none",
        "null",
        "nan"
    ]:
        return None

    return subtitle

metadata["subtitle_clean"] = (
    metadata["subtitle"]
    .apply(clean_subtitle)
)

metadata["has_subtitle"] = (
    metadata["subtitle_clean"]
    .notna()
)

metadata["missing_title"] = (
    metadata["title_clean"].isna()
)

metadata["missing_artist"] = (
    metadata["artist_clean"].isna()
)

metadata["missing_id"] = (
    metadata["id"].isna()
)
metadata["audio_url"]

metadata["is_valid"] = (
    (~metadata["missing_id"])
    &
    (~metadata["missing_title"])
    &
    (~metadata["missing_artist"])
    &
    metadata["duration_valid"]
)
# =========================
# PHÂN LOẠI VALID / INVALID
# =========================

valid = metadata[
    metadata["is_valid"]
].copy()

invalid = metadata[
    ~metadata["is_valid"]
].copy()


def get_invalid_reason(row):

    reasons = []

    if row["missing_id"]:
        reasons.append("missing_id")

    if row["missing_title"]:
        reasons.append("missing_title")

    if row["missing_artist"]:
        reasons.append("missing_artist")

    if not row["duration_valid"]:

        if pd.isna(row["duration"]):
            reasons.append("missing_duration")

        elif row["duration"] < MIN_DURATION:
            reasons.append("too_short")

        elif row["duration"] > MAX_DURATION:
            reasons.append("too_long")

    return ",".join(reasons)


invalid["invalid_reason"] = invalid.apply(
    get_invalid_reason,
    axis=1
)


raw_metadata = metadata.copy()

raw_metadata.to_csv(
    OUTPUT_DIR / "raw_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

raw_metadata.to_parquet(
    OUTPUT_DIR / "raw_metadata.parquet",
    index=False
)


columns_to_keep = [
    "id",

    # dữ liệu gốc quan trọng
    "title",
    "artist",

    # dữ liệu đã clean
    "title_clean",
    "artist_clean",

    # metadata hữu ích
    "duration",
    "view_count",
    "like_count",
    "upload_date",

    # phục vụ lyrics sau này
    "subtitle_clean",
    "has_subtitle"
]


clean_metadata = valid[
    [
        col
        for col in columns_to_keep
        if col in valid.columns
    ]
].copy()


clean_metadata.to_csv(
    OUTPUT_DIR / "clean_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

clean_metadata.to_parquet(
    OUTPUT_DIR / "clean_metadata.parquet",
    index=False
)


invalid_columns = [
    "id",
    "title",
    "artist",
    "duration",
    "invalid_reason"
]

invalid_export = invalid[
    [
        col
        for col in invalid_columns
        if col in invalid.columns
    ]
].copy()

invalid_export.to_csv(
    OUTPUT_DIR / "invalid_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n======================")
print("CLEANING REPORT")
print("======================")

print("Total:", len(metadata))

print("Valid:", len(valid))

print("Invalid:", len(invalid))

print(
    "Missing title:",
    metadata["missing_title"].sum()
)

print(
    "Missing artist:",
    metadata["missing_artist"].sum()
)

print(
    "Invalid duration:",
    (~metadata["duration_valid"]).sum()
)

print(
    "Has subtitle:",
    metadata["has_subtitle"].sum()
)

print("\nFiles created:")
print("Output folder:", OUTPUT_DIR)

print(
    "- raw_metadata.csv:",
    len(raw_metadata),
    "rows"
)

print(
    "- clean_metadata.csv:",
    len(clean_metadata),
    "rows"
)

print(
    "- invalid_metadata.csv:",
    len(invalid_export),
    "rows"
)