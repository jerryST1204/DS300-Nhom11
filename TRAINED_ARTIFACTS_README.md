# Recommendation model artifacts

The main experimental workflow is `Music_Recommendation.ipynb`: it compares
content-based, Funk SVD, item-based CF, rating-aware popularity, and a hybrid
rank fusion on a held-out split. Its bundled interactions are synthetic and
metrics are for technical demonstration only.

`trained_artifacts/` is the separate lightweight CLI bundle rebuilt from the
combined VMusic + `dataViet` catalog. That runtime supports content similarity,
catalog popularity, and a new-user blend of those two rankings; it does not
load the notebook's Funk SVD or item-based CF models.

## Build

From the repository root:

```powershell
python -m pip install -r ials_music_recommendation/requirements.txt
python -m ials_music_recommendation.train_model_artifacts
```

The trainer reads the full cleaned catalog:

`ials_music_recommendation/Dataset/data/lyrics_all_sources_cleaned.csv`

The current file has 5,460 rows, including one `dataViet` row without a usable
title. The catalog loader warns and skips invalid rows, so the trained catalog
contains 5,459 tracks (4,820 VMusic and 639 `dataViet`). Curated lyrics are
available for 2,600 tracks; other tracks remain recommendable by title, artist,
and any available genre metadata.

It keeps the full track catalog and uses lyric text only where
`lyrics_use_for_vector` is true. Titles and artists are available for all valid
catalog rows; genre labels are weighted by their supplied confidence. Captions
are not substituted for lyrics.

The output contains:

- `content_based.joblib`: fitted sparse content vectors and recommendation model.
- `catalog.csv`: app metadata, including source, genre, and VMusic popularity fields.
- `mappings.json`: stable `track_id` to internal `song_id` mapping.
- `manifest.json`: catalog fingerprint, model settings, runtime versions, and checksums.

Genre confidence is qualitative, not a calibrated probability. The feature
weights are high=1.0, medium=0.75, low=0.25, very_low=0.1; missing/unknown
confidence receives no genre weight. Lyrics are included only when approved by
`lyrics_use_for_vector`.

Load artifacts once when the backend starts:

```python
from ials_music_recommendation.model_bundle import TrainedModelBundle

models = TrainedModelBundle.from_directory()
similar = models.recommend_similar(["kXYiU_JCYtU"], top_k=10)
new_user = models.recommend_new_user(
    liked_track_ids=["kXYiU_JCYtU"],
    top_k=10,
)
popular = models.recommend_popular(top_k=10)
hybrid = models.recommend_new_user(
    liked_track_ids=["kXYiU_JCYtU"],
    top_k=10,
    content_weight=0.7,
)
```

The hybrid defaults to `content_weight=0.7` so liked-track similarity has more
influence than the global catalog popularity list. This is a practical demo
setting, not a value tuned against real listener feedback.

Every recommendation includes `track_id`, `title`, `artist`, and `score`; genre
and source fields are included when present. Store `track_id` in the app. The
internal `song_id` is only valid with the exact catalog mapping in this bundle.

Run the command-line demo after building artifacts:

```powershell
python .\main.py --track-id kXYiU_JCYtU --top-k 10
```

Omit `--track-id` to see the catalog popularity list. Popularity uses VMusic
view/like metadata; `dataViet` tracks have no matching counts and therefore do
not appear in that non-personalized list. The hybrid can still return
`dataViet` tracks through content similarity.

## Data limits

- The dataset does not contain real user listening history. The CSV under
  `Dataset/user data/` is synthetic and is not used to fit or validate these
  deployable recommendations.
- Collaborative filtering and Funk SVD are disabled in the current app bundle.
  Collect real user/song interactions before evaluating or enabling them.
- `stats_music.csv` is a separate 28,372-row genre dataset without matching
  catalog track IDs; it is not used by this recommender.
- `score` is a ranking score, not a probability or measured accuracy.
- Some VMusic genres are heuristic; `genre_confidence` is qualitative and is
  not a calibrated probability. The five `dataViet` genre labels come from
  source folders.
- Rebuild artifacts after changing the catalog. Version 1 artifacts from the
  old synthetic-history pipeline are rejected by the current runtime.
- The unused version 1 archive and legacy model files have been removed. The
  current runtime requires only `catalog.csv`, `mappings.json`,
  `content_based.joblib`, and `manifest.json`.
