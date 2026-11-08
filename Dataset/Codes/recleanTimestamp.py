import pandas as pd
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "clean_metadata.csv"
OUTPUT_FILE = DATA_DIR / "clean_metadata_no_timestamp.csv"


def remove_subtitle_timestamps(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Bỏ WEBVTT
    text = re.sub(
        r"(?im)^\s*WEBVTT\s*$",
        "",
        text
    )

    # Bỏ metadata VTT
    text = re.sub(
        r"(?im)^\s*(Kind|Language):.*$",
        "",
        text
    )

    # Bỏ timestamp dạng:
    # 00:01:06.480 --> 00:01:11.200
    text = re.sub(
        r"(?m)^\s*"
        r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}"
        r"\s*-->\s*"
        r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}"
        r".*$",
        "",
        text
    )

    # Bỏ timestamp dạng:
    # 01:06.480 --> 01:11.200
    text = re.sub(
        r"(?m)^\s*"
        r"\d{1,2}:\d{2}[.,]\d{3}"
        r"\s*-->\s*"
        r"\d{1,2}:\d{2}[.,]\d{3}"
        r".*$",
        "",
        text
    )

    # Bỏ dòng chỉ chứa số cue
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    # Chuẩn hóa dòng trống
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


df = pd.read_csv(INPUT_FILE)

if "subtitle_clean" not in df.columns:
    raise ValueError("Không tìm thấy cột subtitle_clean")

df["subtitle_clean"] = df["subtitle_clean"].apply(
    remove_subtitle_timestamps
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Đã lưu:", OUTPUT_FILE)