"""Lightweight, offline search over song metadata and approved lyrics."""

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


_MOODS = {
    "buồn": {
        "buồn", "sad", "cô đơn", "lẻ loi", "thất tình", "lụy tình",
        "đau lòng", "tổn thương", "tan vỡ", "chia tay", "nước mắt",
        "đau khổ", "lạc lõng", "u sầu", "sầu", "ảm đạm", "bi thương",
        "thất vọng", "chán nản", "mất mát",
    },
    "vui": {
        "vui", "happy", "hạnh phúc", "yêu đời", "tươi vui", "rộn ràng",
        "nụ cười", "nhảy múa", "phấn khởi", "lạc quan", "tích cực",
        "tưng bừng", "vui vẻ",
    },
    "chill": {
        "chill", "thư giãn", "relax", "nhẹ nhàng", "bình yên", "êm đềm",
        "thảnh thơi", "lofi", "acoustic", "yên tĩnh", "dịu êm",
        "đêm khuya", "đêm muộn", "nghe đêm", "trước khi ngủ",
    },
    "lãng mạn": {
        "lãng mạn", "romantic", "tình yêu", "yêu thương", "say đắm",
        "ngọt ngào", "hẹn hò", "đôi lứa", "tỏ tình", "người yêu",
    },
    "năng lượng": {
        "năng lượng", "sôi động", "bùng nổ", "mạnh mẽ", "hứng khởi",
        "bốc lửa", "quẩy", "tập gym", "tập luyện", "chạy bộ", "thể thao",
        "energetic", "khuấy động", "nhiệt huyết", "sung sức", "hăng hái", "máu lửa", "cháy"
    },
    "hoài niệm": {
        "hoài niệm", "nostalgia", "ngày xưa", "ký ức", "kỷ niệm",
        "tuổi thơ", "thanh xuân", "hồi ức", "ngày cũ", "retro", "vintage",
        "nhớ về",
    },
    "sâu lắng": {
        "sâu lắng", "lắng đọng", "suy tư", "trầm tư", "chiêm nghiệm",
        "tâm sự", "tự sự", "lặng lẽ", "suy ngẫm",
    },
    "tập trung": {
        "tập trung", "focus", "học bài", "học tập", "làm việc", "đọc sách",
        "nhạc nền", "không lời", "instrumental", "study",
    },
}

_MOOD_ALIASES = {
    term: mood
    for mood, terms in _MOODS.items()
    for term in terms
}

_GENRE_ALIASES = {
    "v-pop": "pop",
    "v pop": "pop",
    "nhạc pop": "pop",
    "pop": "pop",
    "indie": "indie",
    "ballad": "ballad",
    "rap": "rap",
    "hip hop": "hip hop",
    "hip-hop": "hip hop",
    "rock": "rock",
    "bolero": "bolero",
    "nhạc trịnh": "nhac trinh",
    "trữ tình": "tru tinh",
    "cách mạng": "cach mang",
}

_STOP_WORDS = {
    "cho", "toi", "mot", "vai", "bai", "nhac", "nghe",
    "tim", "giup", "voi", "kieu", "cua", "co", "the", "nao",
    "va", "de", "toi", "la", "nhung", "bai", "hat",
}


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value).casefold())
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _approved_lyrics(songs: pd.DataFrame) -> pd.Series:
    lyrics_column = "lyrics" if "lyrics" in songs else "lyrics_clean"
    if lyrics_column not in songs:
        return pd.Series("", index=songs.index, dtype="string")

    lyrics = songs[lyrics_column].fillna("").astype(str)
    if "lyrics_use_for_vector" in songs:
        approved = (
            songs["lyrics_use_for_vector"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.casefold()
            .isin({"true", "1", "yes", "y"})
        )
        lyrics = lyrics.where(approved, "")
    return lyrics


class NaturalLanguageMusicSearch:
    """Search the catalog using mood words and metadata in a free-text query.

    Mood matching expands a small Vietnamese keyword dictionary and searches
    the approved lyrics lexically. It is an explainable demo baseline, not a
    semantic emotion model or an LLM.
    """

    def __init__(self, songs: pd.DataFrame) -> None:
        required = {"song_id", "title", "artist"}
        missing = required - set(songs.columns)
        if missing:
            raise ValueError(f"songs is missing columns: {', '.join(sorted(missing))}")
        if songs.empty:
            raise ValueError("songs must contain at least one row.")
        if songs["song_id"].isna().any() or songs["song_id"].duplicated().any():
            raise ValueError("song_id values must be present and unique.")

        self.songs = songs.reset_index(drop=True).copy()
        for column in ("title", "artist", "genre"):
            if column not in self.songs:
                self.songs[column] = ""
            self.songs[column] = self.songs[column].fillna("").astype(str)

        lyrics = _approved_lyrics(self.songs)
        self.documents = (
            self.songs["title"] + " " + self.songs["artist"] + " "
            + self.songs["genre"] + " " + lyrics
        ).str.strip()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=150_000,
        )
        self.item_vectors = self.vectorizer.fit_transform(self.documents)
        self._normalized_genres = self.songs["genre"].map(_normalize)
        self._normalized_artists = self.songs["artist"].map(_normalize)

    def parse_query(self, query: str) -> dict[str, list[str]]:
        """Return recognized mood, genre, and artist hints from a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query phải là chuỗi không rỗng.")
        normalized_query = _normalize(query)
        moods = sorted(
            {
                mood
                for alias, mood in _MOOD_ALIASES.items()
                if re.search(rf"(?<!\w){re.escape(_normalize(alias))}(?!\w)", normalized_query)
            }
        )
        genres = sorted(
            {
                genre
                for alias, genre in _GENRE_ALIASES.items()
                if re.search(rf"(?<!\w){re.escape(_normalize(alias))}(?!\w)", normalized_query)
            }
        )
        artists = sorted(
            {
                artist.strip()
                for artist, normalized_artist in zip(
                    self.songs["artist"], self._normalized_artists
                )
                if normalized_artist
                and re.search(
                    rf"(?<!\w){re.escape(normalized_artist)}(?!\w)",
                    normalized_query,
                )
            },
            key=lambda artist: (-len(artist), artist.casefold()),
        )
        return {"moods": moods, "genres": genres, "artists": artists}

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Return lexically relevant songs, optionally filtered by genre."""
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0.")
        hints = self.parse_query(query)
        normalized_query = _normalize(query)
        terms = [
            token for token in re.findall(r"[\w-]+", normalized_query)
            if token not in _STOP_WORDS
        ]
        mood_terms = sorted(
            {
                term
                for mood in hints["moods"]
                for term in _MOODS[mood]
            }
        )
        expanded_query = " ".join([*terms, *mood_terms, *hints["genres"], *hints["artists"]])
        if not expanded_query.strip():
            raise ValueError("Không tìm thấy từ khóa để tìm bài hát.")
        query_vector = self.vectorizer.transform([expanded_query])
        scores = (query_vector @ self.item_vectors.T).toarray().ravel()
        if hints["artists"]:
            normalized_artists = {_normalize(artist) for artist in hints["artists"]}
            artist_matches = self._normalized_artists.isin(normalized_artists).to_numpy()
            scores[artist_matches] += 0.2

        eligible = np.ones(len(self.songs), dtype=bool)
        if hints["genres"]:
            genre_matches = np.zeros(len(self.songs), dtype=bool)
            for genre in hints["genres"]:
                genre_matches |= self._normalized_genres.str.contains(
                    re.escape(genre), regex=True, na=False
                ).to_numpy()
            if not genre_matches.any():
                return self.songs.iloc[0:0].assign(score=pd.Series(dtype=float))
            eligible &= genre_matches

        scores[~eligible] = -np.inf
        ranked_rows = np.argsort(-scores, kind="stable")
        ranked_rows = ranked_rows[np.isfinite(scores[ranked_rows])][:top_k]
        result = self.songs.iloc[ranked_rows].copy()
        result["score"] = scores[ranked_rows]
        return result.reset_index(drop=True)
