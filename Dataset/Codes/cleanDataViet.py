from pathlib import Path
import pandas as pd
import re
import struct
import unicodedata

# CẤU HÌNH
DATASET_ROOT = Path(r"D:\0_College\#DS300\Music")

OUTPUT_SEGMENTS = "cleaned_segments.csv"
OUTPUT_TRACKS = "cleaned_tracks_for_lyrics.csv"

# HÀM HỖ TRỢ

def normalize_genre(name):
   
    name = name.strip()
    name = name.lower()
    name = re.sub(r"\s+", "_", name)

    return name


def remove_segment(filename):

    name = Path(filename).stem

    name = re.sub(
        r"_segment[_-]?\d+$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name


def get_segment_index(filename):

    match = re.search(
        r"_segment[_-]?(\d+)",
        filename,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return 1


def remove_download_code(name):

    name = name.rstrip(".")

    # Mã số dài ở cuối, có thể kèm hq/128/320/lossless
    name = re.sub(
        r"[-_]\d{5,}(?:[_-](?:hq|128|320|lossless))?$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Mã alphanumeric ngắn thường được ngăn bằng underscore:
    # _3b9rq_hq, _xcnk, _mv2t, _43cd ...
    name = re.sub(
        r"_[a-z0-9]{3,6}(?:[_-](?:hq|128|320|lossless))?$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Trường hợp chỉ còn hậu tố chất lượng
    name = re.sub(
        r"[_-](?:hq|128|320|lossless)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name

def remove_track_number(name):

    name = re.sub(
        r"^\s*\d{1,3}\s*[-._]*\s*",
        "",
        name
    )

    return name

def split_camel_case(text):

    # XMLParser -> XML Parser
    text = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1 \2",
        text
    )

    # duongTruong -> duong Truong
    text = re.sub(
        r"([a-zà-ỹ])([A-Z])",
        r"\1 \2",
        text
    )

    return text


def clean_track_name(raw_name):

    name = raw_name.strip()

    name = name.rstrip(".")

    # Xóa hậu tố duplicate: (1), (2), ...
    name = re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        name
    )

    # Xóa prefix số
    name = re.sub(
        r"^(?:\s*\d{1,3}\s*[-._]*)+",
        "",
        name
    )

    # Bỏ dấu chấm cuối trước để regex nhận đúng hậu tố download
    name = name.rstrip(".")

    # Bỏ mã download / quality
    name = remove_download_code(name)

    # Bỏ số thứ tự
    name = remove_track_number(name)

    # underscore còn lại -> khoảng trắng
    name = name.replace("_", " ")

    # tách CamelCase/PascalCase
    name = split_camel_case(name)

    # chuẩn hóa dấu gạch ngang
    name = re.sub(r"\s*-\s*", " - ", name)

    # xóa ký tự không cần thiết
    name = re.sub(
        r"[^\wÀ-ỹ\s\-']",
        " ",
        name,
        flags=re.UNICODE
    )

    # chuẩn hóa khoảng trắng
    name = re.sub(r"\s+", " ", name).strip(" -")

    return name


def split_title_artist(clean_name):

    parts = [p.strip() for p in clean_name.split(" - ") if p.strip()]

    if len(parts) == 2:
        title, artist = parts

        # Artist quá ngắn/dài bất thường thì không tự tách
        artist_words = artist.split()
        if 1 <= len(artist_words) <= 6 and len(artist) >= 2:
            return title, artist, False

    if len(parts) > 2:
        return clean_name, "", True

    return clean_name, "", False

def create_title_query(title):

    query = title
    query = query.replace("-", " ")
    query = re.sub(r"\s+", " ", query).strip()

    return query

def slugify(text):

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower()

    text = re.sub(r"[^a-z0-9]+", "-", text)

    return text.strip("-")


def get_audio_info(path):

    try:
        with path.open("rb") as audio_file:
            header = audio_file.read(12)

            if (
                len(header) == 12
                and header[:4] == b"RIFF"
                and header[8:12] == b"WAVE"
            ):
                riff_size = struct.unpack("<I", header[4:8])[0]
                riff_end = riff_size + 8
                file_size = path.stat().st_size

                if riff_end > file_size or riff_end < 12:
                    raise ValueError("Kích thước RIFF không hợp lệ")

                audio_format = None
                sample_rate = None
                byte_rate = None
                channels = None
                data_size = 0
                position = 12

                while position + 8 <= riff_end:
                    audio_file.seek(position)
                    chunk_header = audio_file.read(8)
                    if len(chunk_header) != 8:
                        raise ValueError("Chunk WAV bị thiếu header")

                    chunk_id, chunk_size = struct.unpack(
                        "<4sI",
                        chunk_header
                    )
                    chunk_start = position + 8
                    chunk_end = chunk_start + chunk_size

                    if chunk_end > riff_end:
                        raise ValueError("Chunk WAV vượt quá kích thước RIFF")

                    if chunk_id == b"fmt ":
                        if chunk_size < 16:
                            raise ValueError("Chunk fmt WAV quá ngắn")

                        fmt_data = audio_file.read(16)
                        (
                            audio_format,
                            channels,
                            sample_rate,
                            byte_rate,
                            _,
                            _,
                        ) = struct.unpack("<HHIIHH", fmt_data)
                    elif chunk_id == b"data":
                        data_size += chunk_size

                    position = chunk_end + (chunk_size & 1)

                if (
                    audio_format is None
                    or not sample_rate
                    or not byte_rate
                    or channels is None
                    or data_size == 0
                ):
                    raise ValueError("Thiếu metadata fmt hoặc data trong WAV")

                duration = round(data_size / byte_rate, 3)
                return duration, sample_rate, channels

        try:
            from mutagen.mp3 import MP3
        except ModuleNotFoundError as error:
            if error.name != "mutagen":
                raise
            raise RuntimeError(
                "File không phải WAV; cần cài mutagen để đọc metadata MP3"
            ) from error

        audio = MP3(path)
        duration = round(audio.info.length, 3)

        sample_rate = getattr(
            audio.info,
            "sample_rate",
            None
        )

        channels = getattr(
            audio.info,
            "channels",
            None
        )

        return duration, sample_rate, channels

    except Exception as error:
        print(
            f"[WARNING] Không đọc được metadata: {path} ({error})"
        )

        return None, None, None


def needs_review(clean_name, title="", artist="", ambiguous_hyphen=False):

    if not clean_name:
        return True, "empty_name"

    compact = re.sub(r"[^A-Za-z0-9]", "", clean_name)

    if len(compact) <= 2:
        return True, "too_short"

    if compact.isdigit():
        return True, "numeric_name"

    if re.match(r"^\d", clean_name):
        return True, "starts_with_number"

    if re.search(r"\b\d{5,}\b", clean_name):
        return True, "download_id_remaining"

    if re.search(r"\bhq\b", clean_name, flags=re.IGNORECASE):
        return True, "hq_remaining"

    # Mã kiểu 3b9rq, 43cd, mv2t... còn sót ở cuối
    # if re.search(r"\b(?=[a-z0-9]{3,6}\b)(?=[a-z0-9]*\d)[a-z0-9]+$", clean_name, flags=re.IGNORECASE):
    #     return True, "possible_download_code"

    if ambiguous_hyphen:
        return True, "ambiguous_title_artist"

    if not title:
        return True, "empty_title"

    # Hash hex dài
    if len(compact) >= 12 and re.fullmatch(r"[a-fA-F0-9]+", compact):
        return True, "hash_like_name"

    return False, ""

def scan_dataset():

    rows = []

    audio_files = list(
        DATASET_ROOT.rglob("*.mp3")
    )

    print(
        f"Tìm thấy {len(audio_files)} file có đuôi .mp3"
    )

    for index, path in enumerate(audio_files, 1):

        relative_path = path.relative_to(
            DATASET_ROOT
        )

        # Genre

        if len(relative_path.parts) >= 2:
            genre_folder = relative_path.parts[0]
        else:
            genre_folder = "unknown"

        genre = normalize_genre(
            genre_folder
        )

        # Tên bài gốc

        raw_track_name = remove_segment(
            path.name
        )

        clean_name = clean_track_name(
            raw_track_name
        )

        title, artist, ambiguous_hyphen = split_title_artist(
            clean_name
        )

        title_query = create_title_query(
            title
        )

        segment_index = get_segment_index(
            path.name
        )

        # Audio metadata

        duration, sample_rate, channels = \
            get_audio_info(path)

        file_size = path.stat().st_size

        # Track ID

        raw_id = (
            genre
            + ":"
            + raw_track_name.lower()
        )

        raw_id = re.sub(
            r"\s+",
            "",
            raw_id
        )

        # Review

        review, reason = needs_review(
            clean_name,
            title=title,
            artist=artist,
            ambiguous_hyphen=ambiguous_hyphen
        )

        rows.append({
            "file_path": str(relative_path),
            "genre": genre,
            "track_id": raw_id,
            "track_name": raw_track_name,
            "segment_index": segment_index,
            "file_size_bytes": file_size,
            "duration_seconds": duration,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "clean_name": clean_name,
            "title": title,
            "title_query": title_query,
            "artist": artist,
            "lyrics_status": "missing",
            "review_needed": (
                "yes"
                if review
                else "no"
            ),
            "review_reason": reason
        })

        if index % 500 == 0:
            print(
                f"Đã xử lý {index}/{len(audio_files)}"
            )

    return pd.DataFrame(rows)

# TẠO METADATA CẤP BÀI HÁT

def create_track_metadata(segments_df):

    tracks = []

    grouped = segments_df.groupby(
        "track_id",
        sort=False
    )

    for track_id, group in grouped:

        first = group.iloc[0]

        durations = pd.to_numeric(
            group["duration_seconds"],
            errors="coerce"
        )

        # Tổng thời lượng các segment
        if durations.notna().any():
            total_duration = round(
                durations.sum(),
                2
            )
        else:
            total_duration = None

        tracks.append({

            "track_id":
                track_id,

            "genre":
                first["genre"],

            "raw_track_name":
                first["track_name"],

            "clean_name":
                first["clean_name"],

            "title":
                first["title"],

            "title_query":
                first["title_query"],

            "artist":
                first["artist"],

            # Tổng thời lượng của các segment
            "duration_seconds":
                total_duration,

            "segment_count":
                len(group),

            "total_file_size_bytes":
                group["file_size_bytes"].sum(),

            # để chuẩn bị cho LRCLIB
            "lyrics":
                "",

            "lyrics_status":
                "missing",

            "review_needed":
                first["review_needed"],

            "review_reason":
                first["review_reason"]
        })

    return pd.DataFrame(tracks)

# MAIN

def main():

    if not DATASET_ROOT.exists():

        print(
            "Không tìm thấy thư mục dataset:"
        )

        print(DATASET_ROOT)

        return

    print("=" * 60)
    print("SCAN MUSIC DATASET")
    print("=" * 60)

    segments_df = scan_dataset()

    # Sort segment

    segments_df = segments_df.sort_values(
        [
            "genre",
            "track_id",
            "segment_index"
        ]
    )

    # Tạo track metadata

    tracks_df = create_track_metadata(
        segments_df
    )

    tracks_df = tracks_df.sort_values(
        [
            "genre",
            "clean_name"
        ]
    )

    # Lưu

    segments_df.to_csv(
        OUTPUT_SEGMENTS,
        index=False,
        encoding="utf-8-sig"
    )

    tracks_df.to_csv(
        OUTPUT_TRACKS,
        index=False,
        encoding="utf-8-sig"
    )

    # Thống kê

    print()
    print("=" * 60)
    print("HOÀN TẤT")
    print("=" * 60)

    print(
        f"Segment: {len(segments_df)}"
    )

    print(
        f"Bài hát duy nhất: {len(tracks_df)}"
    )

    print()
    print("Số bài theo genre:")

    print(
        tracks_df["genre"]
        .value_counts()
        .to_string()
    )

    review_count = (
        tracks_df["review_needed"]
        .eq("yes")
        .sum()
    )

    print()
    print(
        f"Cần kiểm tra thủ công: {review_count}"
    )

    print()
    print(
        f"Đã tạo: {OUTPUT_SEGMENTS}"
    )

    print(
        f"Đã tạo: {OUTPUT_TRACKS}"
    )


if __name__ == "__main__":
    main()