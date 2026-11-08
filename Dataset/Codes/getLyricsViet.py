from pathlib import Path
import pandas as pd
import requests
import time
import math
import re
import unicodedata
from difflib import SequenceMatcher


# ==========================================================
# CẤU HÌNH
# ==========================================================

DATA_DIR = Path(__file__).resolve().parent / "dataViet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "cleaned_tracks_for_lyrics.csv"
OUTPUT_CSV = DATA_DIR / "tracks_with_lyrics.csv"

LRCLIB_BASE_URL = "https://lrclib.net"

# LRCLIB yêu cầu client tự nhận diện bằng User-Agent
HEADERS = {
    "User-Agent": "VietnamMusicRecommendation/1.0 (student project)"
}

# Delay giữa các request để tránh rate limit
REQUEST_DELAY = 0.35

# Số lần retry cho lỗi mạng / 429 / 5xx
MAX_RETRIES = 4

# Timeout mỗi request
TIMEOUT = 20

# Lưu checkpoint sau mỗi N bài đã xử lý
SAVE_EVERY = 10

# Nếu True, vẫn thử lấy lyrics cho các bài review_needed=yes.
# Khuyến nghị để True vì nhiều bài review chỉ bị mơ hồ artist nhưng title vẫn dùng được.
PROCESS_REVIEW_NEEDED = True

# Ngưỡng similarity để nhận kết quả search
MIN_TITLE_SCORE = 0.72
MIN_COMBINED_SCORE = 0.70


# ==========================================================
# HÀM CHUẨN HÓA
# ==========================================================

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(text):
    """
    Chuẩn hóa để so sánh title/artist:
    - lower
    - bỏ dấu tiếng Việt
    - bỏ ký tự đặc biệt
    - chuẩn hóa khoảng trắng
    """
    text = safe_text(text).lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # xử lý riêng đ/Đ
    text = text.replace("đ", "d")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ==========================================================
# HTTP / RETRY
# ==========================================================

def lrclib_request(endpoint, params):
    """
    Gọi LRCLIB có retry.
    Trả về:
        ("ok", json_data)
        ("not_found", None)
        ("error", message)
    """

    url = LRCLIB_BASE_URL + endpoint

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT
            )

            # Thành công
            if response.status_code == 200:
                try:
                    return "ok", response.json()
                except ValueError:
                    return "error", "response_not_json"

            # Không tìm thấy exact match
            if response.status_code == 404:
                return "not_found", None

            # Rate limit
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = float(retry_after)
                except (TypeError, ValueError):
                    wait_time = 2.0 * attempt

                print(
                    f"    [429] Rate limit. Chờ {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue

            # Server bận / lỗi server
            if response.status_code in (500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = float(retry_after)
                except (TypeError, ValueError):
                    wait_time = 1.5 * attempt

                print(
                    f"    [{response.status_code}] Server bận. "
                    f"Retry sau {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue

            return (
                "error",
                f"http_{response.status_code}: {response.text[:200]}"
            )

        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                return "error", f"network_error: {error}"

            wait_time = 1.5 * attempt

            print(
                f"    [NETWORK] {error}. Retry sau {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

    return "error", "max_retries_exceeded"


# ==========================================================
# LRCLIB MATCHING
# ==========================================================

def extract_plain_lyrics(record):
    """
    Chỉ lấy plainLyrics.
    Không lấy syncedLyrics.
    """

    if not isinstance(record, dict):
        return ""

    if record.get("instrumental") is True:
        return ""

    lyrics = record.get("plainLyrics")

    if lyrics is None:
        return ""

    lyrics = str(lyrics).strip()

    return lyrics


def exact_get(title, artist, duration=None):
    """
    /api/get yêu cầu title + artist.
    Không truyền duration mặc định vì duration trong CSV là tổng các segment
    và có thể không trùng chính xác record LRCLIB.
    """

    title = safe_text(title)
    artist = safe_text(artist)

    if not title or not artist:
        return None, "skipped_no_artist"

    params = {
        "track_name": title,
        "artist_name": artist
    }

    # Chỉ dùng duration nếu thật sự hợp lệ và muốn bật sau.
    # LRCLIB match duration khá chặt, nên ở dataset này KHÔNG truyền.
    #
    # if duration is not None and 1 <= duration <= 3600:
    #     params["duration"] = round(float(duration))

    status, data = lrclib_request(
        "/api/get",
        params
    )

    time.sleep(REQUEST_DELAY)

    if status == "ok":
        lyrics = extract_plain_lyrics(data)

        if lyrics:
            return data, "exact"

        if isinstance(data, dict) and data.get("instrumental") is True:
            return data, "instrumental"

        return None, "exact_no_plain_lyrics"

    if status == "not_found":
        return None, "not_found"

    return None, status


def search_lrclib(title, artist=""):
    """
    Search có cấu trúc.
    Nếu có artist:
        track_name + artist_name
    Nếu không:
        track_name
    """

    title = safe_text(title)
    artist = safe_text(artist)

    if not title:
        return [], "empty_title"

    params = {
        "track_name": title
    }

    if artist:
        params["artist_name"] = artist

    status, data = lrclib_request(
        "/api/search",
        params
    )

    time.sleep(REQUEST_DELAY)

    if status != "ok":
        return [], status

    if not isinstance(data, list):
        return [], "invalid_search_response"

    return data, "search"


def search_free_text(query):
    """
    Fallback cuối bằng q.
    """

    query = safe_text(query)

    if not query:
        return [], "empty_query"

    status, data = lrclib_request(
        "/api/search",
        {"q": query}
    )

    time.sleep(REQUEST_DELAY)

    if status != "ok":
        return [], status

    if not isinstance(data, list):
        return [], "invalid_search_response"

    return data, "search_q"


def score_result(result, title, artist=""):
    """
    Chấm điểm để tránh lấy bừa kết quả đầu tiên.
    """

    result_title = safe_text(
        result.get("trackName") or result.get("name")
    )
    result_artist = safe_text(
        result.get("artistName")
    )

    title_score = similarity(title, result_title)

    if artist:
        artist_score = similarity(artist, result_artist)

        # ưu tiên title hơn artist một chút
        combined = 0.72 * title_score + 0.28 * artist_score
    else:
        artist_score = 0.0
        combined = title_score

    return combined, title_score, artist_score


def choose_best_result(results, title, artist=""):
    """
    Chọn record tốt nhất có plainLyrics.
    """

    candidates = []

    for result in results:
        if not isinstance(result, dict):
            continue

        lyrics = extract_plain_lyrics(result)

        # Chỉ lấy record có plain lyrics
        if not lyrics:
            continue

        combined, title_score, artist_score = score_result(
            result,
            title,
            artist
        )

        candidates.append(
            (
                combined,
                title_score,
                artist_score,
                result
            )
        )

    if not candidates:
        return None, 0.0, 0.0, 0.0

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2]
        ),
        reverse=True
    )

    combined, title_score, artist_score, best = candidates[0]

    # title phải đủ giống
    if title_score < MIN_TITLE_SCORE:
        return None, combined, title_score, artist_score

    # nếu có artist thì combined cũng phải đủ tốt
    if artist and combined < MIN_COMBINED_SCORE:
        return None, combined, title_score, artist_score

    return best, combined, title_score, artist_score


# ==========================================================
# PIPELINE MỖI BÀI
# ==========================================================

def get_lyrics_for_row(row):
    """
    Thứ tự:
    1. exact title + artist
    2. search title + artist
    3. search title
    4. search title_query
    5. q = title + artist
    6. q = title_query
    """

    title = safe_text(row.get("title"))
    title_query = safe_text(row.get("title_query"))
    artist = safe_text(row.get("artist"))

    if not title:
        return {
            "lyrics": "",
            "lyrics_status": "missing",
            "lyrics_source": "",
            "matched_title": "",
            "matched_artist": "",
            "lrclib_id": "",
            "match_score": "",
            "lyrics_error": "empty_title"
        }

    # ------------------------------------------------------
    # 1. Exact: title + artist
    # ------------------------------------------------------

    if artist:
        record, source = exact_get(
            title,
            artist
        )

        if record is not None:
            lyrics = extract_plain_lyrics(record)

            if lyrics:
                return build_success(
                    record,
                    lyrics,
                    source,
                    1.0
                )

            if source == "instrumental":
                return {
                    "lyrics": "",
                    "lyrics_status": "instrumental",
                    "lyrics_source": "lrclib_exact",
                    "matched_title": safe_text(record.get("trackName")),
                    "matched_artist": safe_text(record.get("artistName")),
                    "lrclib_id": safe_text(record.get("id")),
                    "match_score": "1.0",
                    "lyrics_error": ""
                }

    # ------------------------------------------------------
    # 2. Search title + artist
    # ------------------------------------------------------

    if artist:
        results, _ = search_lrclib(
            title,
            artist
        )

        best, score, _, _ = choose_best_result(
            results,
            title,
            artist
        )

        if best:
            return build_success(
                best,
                extract_plain_lyrics(best),
                "lrclib_search_title_artist",
                score
            )

    # ------------------------------------------------------
    # 3. Search chỉ title
    # ------------------------------------------------------

    results, _ = search_lrclib(
        title
    )

    best, score, _, _ = choose_best_result(
        results,
        title,
        ""
    )

    if best:
        return build_success(
            best,
            extract_plain_lyrics(best),
            "lrclib_search_title",
            score
        )

    # ------------------------------------------------------
    # 4. Search title_query nếu khác title
    # ------------------------------------------------------

    if (
        title_query
        and normalize_text(title_query) != normalize_text(title)
    ):
        results, _ = search_lrclib(
            title_query
        )

        best, score, _, _ = choose_best_result(
            results,
            title_query,
            ""
        )

        if best:
            return build_success(
                best,
                extract_plain_lyrics(best),
                "lrclib_search_title_query",
                score
            )

    # ------------------------------------------------------
    # 5. q = title + artist
    # ------------------------------------------------------

    if artist:
        query = f"{title} {artist}".strip()

        results, _ = search_free_text(query)

        best, score, _, _ = choose_best_result(
            results,
            title,
            artist
        )

        if best:
            return build_success(
                best,
                extract_plain_lyrics(best),
                "lrclib_q_title_artist",
                score
            )

    # ------------------------------------------------------
    # 6. q = title_query/title
    # ------------------------------------------------------

    fallback_query = title_query or title

    results, status = search_free_text(
        fallback_query
    )

    best, score, _, _ = choose_best_result(
        results,
        title,
        artist
    )

    if best:
        return build_success(
            best,
            extract_plain_lyrics(best),
            "lrclib_q_fallback",
            score
        )

    return {
        "lyrics": "",
        "lyrics_status": "missing",
        "lyrics_source": "",
        "matched_title": "",
        "matched_artist": "",
        "lrclib_id": "",
        "match_score": "",
        "lyrics_error": (
            ""
            if status in ("ok", "search_q")
            else status
        )
    }


def build_success(record, lyrics, source, score):
    return {
        "lyrics": lyrics,
        "lyrics_status": "found",
        "lyrics_source": source,
        "matched_title": safe_text(
            record.get("trackName") or record.get("name")
        ),
        "matched_artist": safe_text(
            record.get("artistName")
        ),
        "lrclib_id": safe_text(record.get("id")),
        "match_score": str(round(float(score), 4)),
        "lyrics_error": ""
    }


# ==========================================================
# CHECKPOINT / FILE
# ==========================================================

def ensure_columns(df):
    defaults = {
        "lyrics": "",
        "lyrics_status": "missing",
        "lyrics_source": "",
        "matched_title": "",
        "matched_artist": "",
        "lrclib_id": "",
        "match_score": "",
        "lyrics_error": ""
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    # tránh NaN gây rắc rối
    for column in defaults:
        df[column] = df[column].fillna("")

    return df


def save_checkpoint(df):
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

# MAIN

def main():
    # Nếu đã có output thì tiếp tục từ checkpoint
    if OUTPUT_CSV.exists():
        print(f"Tiếp tục từ checkpoint: {OUTPUT_CSV}")
        df = pd.read_csv(
            OUTPUT_CSV,
            encoding="utf-8-sig"
        )
    else:
        if not INPUT_CSV.exists():
            print("Không tìm thấy input:")
            print(INPUT_CSV)
            return

        print(f"Đọc input: {INPUT_CSV}")
        df = pd.read_csv(
            INPUT_CSV,
            encoding="utf-8-sig"
        )

    df = ensure_columns(df)

    required = [
        "track_id",
        "title",
        "title_query",
        "artist"
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        print(
            "Thiếu các cột bắt buộc:",
            ", ".join(missing_columns)
        )
        return

    total = len(df)

    found_before = (
        df["lyrics_status"]
        .astype(str)
        .str.lower()
        .eq("found")
        .sum()
    )

    print("=" * 70)
    print("GET LYRICS FROM LRCLIB")
    print("=" * 70)
    print(f"Tổng số bài : {total}")
    print(f"Đã có lyrics: {found_before}")
    print()

    processed_this_run = 0

    try:
        for index, row in df.iterrows():
            current_status = safe_text(
                row.get("lyrics_status")
            ).lower()

            current_lyrics = safe_text(
                row.get("lyrics")
            )

            # Đã có lyrics -> bỏ qua
            if current_status == "found" and current_lyrics:
                continue

            # Instrumental đã xác định -> bỏ qua
            if current_status == "instrumental":
                continue

            review_needed = safe_text(
                row.get("review_needed")
            ).lower()

            if (
                not PROCESS_REVIEW_NEEDED
                and review_needed == "yes"
            ):
                continue

            title = safe_text(row.get("title"))
            artist = safe_text(row.get("artist"))

            print(
                f"[{index + 1}/{total}] "
                f"{title}"
                + (f" - {artist}" if artist else "")
            )

            result = get_lyrics_for_row(row)

            for key, value in result.items():
                df.at[index, key] = value

            processed_this_run += 1

            if result["lyrics_status"] == "found":
                print(
                    f"    ✓ FOUND | "
                    f"{result['matched_title']} - "
                    f"{result['matched_artist']} | "
                    f"{result['lyrics_source']} | "
                    f"score={result['match_score']}"
                )
            elif result["lyrics_status"] == "instrumental":
                print("    ♪ Instrumental")
            else:
                print("    ✗ Missing")

            # checkpoint
            if processed_this_run % SAVE_EVERY == 0:
                save_checkpoint(df)
                print(
                    f"    [checkpoint] Đã lưu {OUTPUT_CSV}"
                )

    except KeyboardInterrupt:
        print()
        print("Đã dừng bằng Ctrl+C. Đang lưu checkpoint...")
        save_checkpoint(df)
        print(f"Đã lưu: {OUTPUT_CSV}")
        return

    # lưu cuối
    save_checkpoint(df)

    found = (
        df["lyrics_status"]
        .astype(str)
        .str.lower()
        .eq("found")
        .sum()
    )

    instrumental = (
        df["lyrics_status"]
        .astype(str)
        .str.lower()
        .eq("instrumental")
        .sum()
    )

    missing = (
        df["lyrics_status"]
        .astype(str)
        .str.lower()
        .eq("missing")
        .sum()
    )

    print()
    print("=" * 70)
    print("HOÀN TẤT")
    print("=" * 70)
    print(f"Tổng bài      : {total}")
    print(f"Có lyrics     : {found}")
    print(f"Instrumental  : {instrumental}")
    print(f"Không tìm thấy: {missing}")
    print()
    print(f"Đã lưu: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
