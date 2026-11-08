"""Add auditable, rule-based primary genre guesses to the song catalog.

Lyrics alone cannot reliably identify a music genre. The classifier therefore
uses explicit genre tags in titles and a small artist/channel profile map first;
lyric language is only a low-confidence fallback.
"""

from __future__ import annotations

import re
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd

if __package__:
    from .data_generator import DEFAULT_CATALOG
else:
    from data_generator import DEFAULT_CATALOG

CATALOG = DEFAULT_CATALOG
OUTPUT_CATALOG = CATALOG.with_name("lyrics_all_sources_with_genres.csv")
UNKNOWN = "Other/Unknown"


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.replace("đ", "d")).strip()


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


# Matched against title + channel/artist metadata; keep aliases distinctive.
ARTIST_PROFILES: dict[str, tuple[str, str]] = {
    # Vietnamese rap and hip-hop
    "den vau": ("Rap/Hip-Hop", "artist_profile"),
    "karik": ("Rap/Hip-Hop", "artist_profile"),
    "binz": ("Rap/Hip-Hop", "artist_profile"),
    "hieuthuhai": ("Rap/Hip-Hop", "artist_profile"),
    "hieu thu hai": ("Rap/Hip-Hop", "artist_profile"),
    "suboi": ("Rap/Hip-Hop", "artist_profile"),
    "wowy": ("Rap/Hip-Hop", "artist_profile"),
    "b ray": ("Rap/Hip-Hop", "artist_profile"),
    "bray": ("Rap/Hip-Hop", "artist_profile"),
    "low g": ("Rap/Hip-Hop", "artist_profile"),
    "g ducky": ("Rap/Hip-Hop", "artist_profile"),
    "rhymastic": ("Rap/Hip-Hop", "artist_profile"),
    "mck": ("Rap/Hip-Hop", "artist_profile"),
    "tlinh": ("Rap/Hip-Hop", "artist_profile"),
    "andree": ("Rap/Hip-Hop", "artist_profile"),
    "bigdaddy": ("Rap/Hip-Hop", "artist_profile"),
    "ricky star": ("Rap/Hip-Hop", "artist_profile"),
    "24k right": ("Rap/Hip-Hop", "artist_profile"),
    "hurrykng": ("Rap/Hip-Hop", "artist_profile"),
    "wean": ("Rap/Hip-Hop", "artist_profile"),
    "justatee": ("Rap/Hip-Hop", "artist_profile"),
    "justa tee": ("Rap/Hip-Hop", "artist_profile"),
    "da lab": ("Rap/Hip-Hop", "artist_profile"),
    # Vietnamese bolero and traditional music
    "phuong anh bolero": ("Bolero", "artist_profile"),
    "quang le": ("Bolero", "artist_profile"),
    "phi nhung": ("Bolero", "artist_profile"),
    "dan nguyen": ("Bolero", "artist_profile"),
    "nhi quynh": ("Bolero", "artist_profile"),
    "manh quynh": ("Bolero", "artist_profile"),
    "huong lan": ("Bolero", "artist_profile"),
    "le quyen": ("Bolero", "artist_profile"),
    "nhac cach mang tien chien": ("Traditional/Folk", "publisher_profile"),
    "trong tan": ("Traditional/Folk", "artist_profile"),
    "cam ly": ("Traditional/Folk", "artist_profile"),
    # Vietnamese pop (individual acts; labels are handled separately below)
    "son tung m tp": ("V-Pop", "artist_profile"),
    "son tung": ("V-Pop", "artist_profile"),
    "mr siro": ("V-Pop", "artist_profile"),
    "trung quan": ("V-Pop", "artist_profile"),
    "soobin": ("V-Pop", "artist_profile"),
    "hoang thuy linh": ("V-Pop", "artist_profile"),
    "vu official": ("V-Pop", "artist_profile"),
    "vu cat tuong": ("V-Pop", "artist_profile"),
    "noo phuoc thinh": ("V-Pop", "artist_profile"),
    "bich phuong": ("V-Pop", "artist_profile"),
    "onlyc": ("V-Pop", "artist_profile"),
    "duc phuc": ("V-Pop", "artist_profile"),
    "truc nhan": ("V-Pop", "artist_profile"),
    "phan manh quynh": ("V-Pop", "artist_profile"),
    "toc tien": ("V-Pop", "artist_profile"),
    "my tam": ("V-Pop", "artist_profile"),
    "amee": ("V-Pop", "artist_profile"),
    "dong nhi": ("V-Pop", "artist_profile"),
    "hoa minzy": ("V-Pop", "artist_profile"),
    "chi pu": ("V-Pop", "artist_profile"),
    "min official": ("V-Pop", "artist_profile"),
    "erik official": ("V-Pop", "artist_profile"),
    "khac viet": ("V-Pop", "artist_profile"),
    "buitruonglinh": ("V-Pop", "artist_profile"),
    "tang duy tan": ("V-Pop", "artist_profile"),
    "thinh suy": ("V-Pop", "artist_profile"),
    "trinh thang binh": ("V-Pop", "artist_profile"),
    "huong tram": ("V-Pop", "artist_profile"),
    "van mai huong": ("V-Pop", "artist_profile"),
    "bao anh": ("V-Pop", "artist_profile"),
    "phuong ly": ("V-Pop", "artist_profile"),
    "andiez": ("V-Pop", "artist_profile"),
    "han sara": ("V-Pop", "artist_profile"),
    "lou hoang": ("V-Pop", "artist_profile"),
    "hoang dung": ("V-Pop", "artist_profile"),
    "tien tien": ("V-Pop", "artist_profile"),
    "miu le": ("V-Pop", "artist_profile"),
    "nguyen ha": ("V-Pop", "artist_profile"),
    "quang hung masterd": ("V-Pop", "artist_profile"),
    # Rock and metal acts
    "linkin park": ("Rock/Metal", "artist_profile"),
    "nirvana": ("Rock/Metal", "artist_profile"),
    "queen": ("Rock/Metal", "artist_profile"),
    "guns n roses": ("Rock/Metal", "artist_profile"),
    "the cranberries": ("Rock/Metal", "artist_profile"),
    "ac dc": ("Rock/Metal", "artist_profile"),
    "evanescence": ("Rock/Metal", "artist_profile"),
    "metallica": ("Rock/Metal", "artist_profile"),
    "arctic monkeys": ("Rock/Metal", "artist_profile"),
    "the verve": ("Rock/Metal", "artist_profile"),
    "green day": ("Rock/Metal", "artist_profile"),
    "buc tuong": ("Rock/Metal", "artist_profile"),
    "microwave band": ("Rock/Metal", "artist_profile"),
    "ngot": ("Rock/Metal", "artist_profile"),
    "chillies": ("Rock/Metal", "artist_profile"),
    # Electronic music and DJ-led artists
    "alan walker": ("Electronic/Dance", "artist_profile"),
    "avicii": ("Electronic/Dance", "artist_profile"),
    "david guetta": ("Electronic/Dance", "artist_profile"),
    "calvin harris": ("Electronic/Dance", "artist_profile"),
    "martin garrix": ("Electronic/Dance", "artist_profile"),
    "marshmello": ("Electronic/Dance", "artist_profile"),
    "the chainsmokers": ("Electronic/Dance", "artist_profile"),
    "breaking copyright": ("Instrumental/Ambient", "publisher_profile"),
    "hearwego": ("Instrumental/Ambient", "publisher_profile"),
    "we are diamond": ("Instrumental/Ambient", "publisher_profile"),
    "infraction": ("Instrumental/Ambient", "publisher_profile"),
    # Other recognizable international genres
    "bts": ("K-Pop", "artist_profile"),
    "blackpink": ("K-Pop", "artist_profile"),
    "twice": ("K-Pop", "artist_profile"),
    "stray kids": ("K-Pop", "artist_profile"),
    "iu": ("K-Pop", "artist_profile"),
    "bad bunny": ("Latin", "artist_profile"),
    "shakira": ("Latin", "artist_profile"),
    "daddy yankee": ("Latin", "artist_profile"),
    "luis fonsi": ("Latin", "artist_profile"),
    "morgan wallen": ("Country/Folk", "artist_profile"),
    "luke bryan": ("Country/Folk", "artist_profile"),
    "johnny cash": ("Country/Folk", "artist_profile"),
    "adele": ("Pop/R&B", "artist_profile"),
    "ariana grande": ("Pop/R&B", "artist_profile"),
    "bruno mars": ("Pop/R&B", "artist_profile"),
    "the weeknd": ("Pop/R&B", "artist_profile"),
    "justin bieber": ("Pop/R&B", "artist_profile"),
    "ed sheeran": ("Pop/R&B", "artist_profile"),
    "taylor swift": ("Pop/R&B", "artist_profile"),
    "drake": ("Rap/Hip-Hop", "artist_profile"),
    "eminem": ("Rap/Hip-Hop", "artist_profile"),
    "kanye west": ("Rap/Hip-Hop", "artist_profile"),
    "50 cent": ("Rap/Hip-Hop", "artist_profile"),
    "nicki minaj": ("Rap/Hip-Hop", "artist_profile"),
    "the notorious big": ("Rap/Hip-Hop", "artist_profile"),
    "notorious big": ("Rap/Hip-Hop", "artist_profile"),
    "notorious b i g": ("Rap/Hip-Hop", "artist_profile"),
    "macklemore": ("Rap/Hip-Hop", "artist_profile"),
    "post malone": ("Pop/R&B", "artist_profile"),
    "akon": ("Pop/R&B", "artist_profile"),
    "hardwell": ("Electronic/Dance", "artist_profile"),
    "the men": ("V-Pop", "artist_profile"),
    "thuy tien": ("V-Pop", "artist_profile"),
    "le bao binh": ("V-Pop", "artist_profile"),
    "nguyen huong ly": ("V-Pop", "artist_profile"),
    "chau khai phong": ("V-Pop", "artist_profile"),
    "thuy chi": ("V-Pop", "artist_profile"),
    "orange": ("V-Pop", "artist_profile"),
    "studio68": ("V-Pop", "publisher_profile"),
    "acv entertainment": ("V-Pop", "publisher_profile"),
    "yeah1 music": ("V-Pop", "publisher_profile"),
    "pops music": ("V-Pop", "publisher_profile"),
    "vie channel music": ("V-Pop", "publisher_profile"),
    "zing mp3": ("V-Pop", "publisher_profile"),
    "st 319 entertainment": ("V-Pop", "publisher_profile"),
    "icm entertainment": ("V-Pop", "publisher_profile"),
    "masew": ("Electronic/Dance", "artist_profile"),
    "orinn mix": ("Electronic/Dance", "publisher_profile"),
    "nhac vang tru tinh": ("Bolero", "publisher_profile"),
    "thuy nga": ("Bolero", "publisher_profile"),
    "breakingcopyright": ("Instrumental/Ambient", "publisher_profile"),
    # Additional frequent international artists in this catalog
    "red hot chili peppers": ("Rock/Metal", "artist_profile"),
    "bon jovi": ("Rock/Metal", "artist_profile"),
    "bryan adams": ("Rock/Metal", "artist_profile"),
    "aerosmith": ("Rock/Metal", "artist_profile"),
    "rolling stones": ("Rock/Metal", "artist_profile"),
    "the rolling stones": ("Rock/Metal", "artist_profile"),
    "luke combs": ("Country/Folk", "artist_profile"),
    "alan jackson": ("Country/Folk", "artist_profile"),
    "shania twain": ("Country/Folk", "artist_profile"),
    "blake shelton": ("Country/Folk", "artist_profile"),
    "chris lane": ("Country/Folk", "artist_profile"),
    "billy ray cyrus": ("Country/Folk", "artist_profile"),
    "willie nelson": ("Country/Folk", "artist_profile"),
    "florida georgia line": ("Country/Folk", "artist_profile"),
    "sabrina carpenter": ("Pop/R&B", "artist_profile"),
    "benson boone": ("Pop/R&B", "artist_profile"),
    "dean lewis": ("Pop/R&B", "artist_profile"),
    "tate mcrae": ("Pop/R&B", "artist_profile"),
    "miley cyrus": ("Pop/R&B", "artist_profile"),
    "sam smith": ("Pop/R&B", "artist_profile"),
    "lady gaga": ("Pop/R&B", "artist_profile"),
    "dua lipa": ("Pop/R&B", "artist_profile"),
    "james arthur": ("Pop/R&B", "artist_profile"),
    "sza": ("R&B/Soul", "artist_profile"),
    "chris brown": ("R&B/Soul", "artist_profile"),
    "khalid": ("R&B/Soul", "artist_profile"),
    "coco jones": ("R&B/Soul", "artist_profile"),
    "victoria monet": ("R&B/Soul", "artist_profile"),
    "muni long": ("R&B/Soul", "artist_profile"),
    "nelly": ("Rap/Hip-Hop", "artist_profile"),
    "dr dre": ("Rap/Hip-Hop", "artist_profile"),
    "migos": ("Rap/Hip-Hop", "artist_profile"),
    "fugees": ("Rap/Hip-Hop", "artist_profile"),
    "powfu": ("Rap/Hip-Hop", "artist_profile"),
    "black eyed peas": ("Pop/R&B", "artist_profile"),
    "phung khanh linh": ("V-Pop", "artist_profile"),
    "mlee": ("V-Pop", "artist_profile"),
    "minh hang": ("V-Pop", "artist_profile"),
    "trinh dinh quang": ("V-Pop", "artist_profile"),
    "jaykii": ("V-Pop", "artist_profile"),
    "meens": ("V-Pop", "artist_profile"),
    "thanh hung": ("V-Pop", "artist_profile"),
    "ho ngoc ha": ("V-Pop", "artist_profile"),
    "tuan hung": ("V-Pop", "artist_profile"),
    "hari won": ("V-Pop", "artist_profile"),
    "datg": ("V-Pop", "artist_profile"),
    "long non la": ("Rap/Hip-Hop", "artist_profile"),
    "7clouds": ("Instrumental/Ambient", "publisher_profile"),
    "orinn sc": ("Electronic/Dance", "publisher_profile"),
    "cyril": ("Electronic/Dance", "artist_profile"),
    "oxlade": ("Afrobeats", "artist_profile"),
    "tyla": ("Afrobeats", "artist_profile"),
}

# Longer profile phrases must win over shorter aliases (for example, "Queen").
SORTED_PROFILES = sorted(ARTIST_PROFILES.items(), key=lambda item: len(item[0]), reverse=True)

TITLE_GENRES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), genre)
    for pattern, genre in (
        (r"\b(?:bolero|nhac vang|tru tinh)\b", "Bolero"),
        (r"\b(?:dan ca|quan ho|cai luong|vi dam|nhac cach mang|tien chien|co nhac)\b", "Traditional/Folk"),
        (r"\b(?:rap viet|hip ?hop|hiphop|rapper|rap)\b", "Rap/Hip-Hop"),
        (r"\b(?:heavy metal|nu metal|metal|punk rock|grunge|alternative rock|rock)\b", "Rock/Metal"),
        (r"\b(?:edm|house|techno|trance|dubstep|electronic dance|dance music)\b", "Electronic/Dance"),
        (r"\b(?:instrumental|ambient|lo ?fi|background music|no copyright music|piano solo|guitar instrumental|soundtrack)\b", "Instrumental/Ambient"),
        (r"\b(?:k pop|kpop)\b", "K-Pop"),
        (r"\b(?:reggaeton|salsa|bachata|latin music)\b", "Latin"),
        (r"\b(?:country music|country song)\b", "Country/Folk"),
        (r"\b(?:classical|orchestral|symphonic)\b", "Classical/Orchestral"),
        (r"\b(?:jazz|blues)\b", "Jazz/Blues"),
        (r"\b(?:afrobeats|afrobeat|amapiano)\b", "Afrobeats"),
    )
)

VIETNAMESE_MARKS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ]|̀|́|̃|̉|̣")


def _has_vietnamese_text(value: object) -> bool:
    text = "" if value is None or pd.isna(value) else str(value)
    return len(VIETNAMESE_MARKS.findall(text)) >= 3


def classify_track(title: object, artist: object, lyrics: object) -> tuple[str, str, str]:
    """Return genre, qualitative confidence, and inference method."""
    title_text = _normalize(title)
    artist_text = _normalize(artist)
    combined = f"{title_text} {artist_text}"

    for pattern, genre in TITLE_GENRES:
        if pattern.search(title_text):
            return genre, "high", "explicit_title_tag"

    for alias, (genre, source) in SORTED_PROFILES:
        title_alias = re.search(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", title_text
        )
        if alias in artist_text or title_alias:
            return genre, "medium", source

    if _has_vietnamese_text(title):
        return "V-Pop", "low", "vietnamese_title_language_fallback"

    normalized_lyrics = (
        "" if lyrics is None or pd.isna(lyrics) else str(lyrics).strip()
    )
    if normalized_lyrics and _has_vietnamese_text(normalized_lyrics):
        return "V-Pop", "low", "vietnamese_language_fallback"

    # Keep unclassifiable instrumental/unknown tracks explicit instead of
    # assigning a genre from weak title or artist clues.
    return UNKNOWN, "low", "no_reliable_text_evidence"


def add_genres(
    csv_path: str | Path = CATALOG,
    output_path: str | Path = OUTPUT_CATALOG,
) -> pd.DataFrame:
    """Fill missing genre labels and write a separate derived catalog.

    Existing `dataViet` folder labels and any genre metadata already attached
    by the data manager are preserved. The source CSV is never overwritten.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    songs = pd.read_csv(csv_path, encoding="utf-8-sig")
    title = songs.get("title_clean", songs.get("title"))
    artist = songs.get("artist_clean", songs.get("artist"))
    if title is None or artist is None:
        missing = sorted({"title", "artist"} - set(songs.columns))
        raise ValueError(f"Catalog is missing required columns: {missing}")
    title = title.fillna("")
    artist = artist.fillna("")
    lyrics = songs.get("lyrics_clean", songs.get("lyrics", pd.Series("", index=songs.index)))
    lyrics = lyrics.fillna("").astype(str)
    if "lyrics_use_for_vector" in songs:
        approved = _truthy(songs["lyrics_use_for_vector"])
        lyrics = lyrics.where(approved, "")

    required = {"title", "artist"}
    missing = required - set(songs.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")

    classified = songs.apply(
        lambda row: classify_track(title.loc[row.name], artist.loc[row.name], lyrics.loc[row.name]),
        axis=1,
        result_type="expand",
    )
    classified.columns = ["genre", "genre_confidence", "genre_source"]
    if "genre" not in songs:
        songs["genre"] = ""
    current_genre = songs["genre"].fillna("").astype(str).str.strip()
    missing_genre = current_genre.eq("")
    songs.loc[missing_genre, "genre"] = classified.loc[missing_genre, "genre"]
    for column in ("genre_confidence", "genre_source"):
        if column not in songs:
            songs[column] = ""
        current = songs[column].fillna("").astype(str).str.strip()
        missing_metadata = current.eq("")
        if column == "genre_confidence":
            # Dataset-provided folder genres are explicit labels; inferred
            # labels use the classifier confidence only where genre was empty.
            explicit = (~missing_genre) & songs.get(
                "dataset_source", pd.Series("", index=songs.index)
            ).astype(str).str.casefold().eq("dataviet")
            songs.loc[missing_metadata & explicit, column] = "high"
            use_inferred = missing_genre & missing_metadata
        elif column == "genre_source":
            explicit = (~missing_genre) & songs.get(
                "dataset_source", pd.Series("", index=songs.index)
            ).astype(str).str.casefold().eq("dataviet")
            songs.loc[missing_metadata & explicit, column] = "dataset_folder_label"
            use_inferred = missing_genre & missing_metadata
        songs.loc[use_inferred, column] = classified.loc[use_inferred, column]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temp_file:
            temp_name = temp_file.name
            songs.to_csv(temp_file, index=False)
        Path(temp_name).replace(output_path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return songs


def main() -> None:
    songs = add_genres()
    print("Genre counts:")
    print(songs["genre"].value_counts(dropna=False).to_string())
    print("\nConfidence counts:")
    print(songs["genre_confidence"].value_counts(dropna=False).to_string())
    print(f"\nWrote derived catalog: {OUTPUT_CATALOG}")


if __name__ == "__main__":
    main()
