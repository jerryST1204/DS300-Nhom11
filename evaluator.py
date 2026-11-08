"""
evaluator.py
────────────
Đánh giá hệ khuyến nghị với các metric phổ biến:
  - Precision@K
  - Recall@K
  - F1@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
  - Hit Rate@K

Sử dụng leave-one-out hoặc train/test split.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Callable


# ─────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────

def train_test_split_interactions(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách dữ liệu train/test theo từng user.

    Với mỗi user, lấy `test_ratio` phần trăm bài đã nghe
    làm ground truth test (giữ lại để đánh giá).

    Parameters
    ----------
    min_interactions : int
        User cần ít nhất bao nhiêu bài mới được đưa vào test.
    """
    rng = np.random.default_rng(random_state)
    train_rows, test_rows = [], []

    for user_id, group in interactions_df.groupby("user_id"):
        if len(group) < min_interactions:
            train_rows.append(group)
            continue

        n_test = max(1, int(len(group) * test_ratio))
        test_idx = rng.choice(group.index, size=n_test, replace=False)
        train_idx = group.index.difference(test_idx)

        train_rows.append(group.loc[train_idx])
        test_rows.append(group.loc[test_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"[Split] Users in test: {test_df['user_id'].nunique()}")
    return train_df, test_df


def df_to_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    value_col: str = "play_count",
) -> sparse.csr_matrix:
    """Convert interactions to a compact CSR user-item matrix."""
    required = {"user_id", "song_id", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {', '.join(sorted(missing))}.")
    if n_users < 1 or n_items < 1:
        raise ValueError("n_users and n_items must both be greater than zero.")

    try:
        users = pd.to_numeric(df["user_id"], errors="raise").to_numpy(dtype=np.float64)
        items = pd.to_numeric(df["song_id"], errors="raise").to_numpy(dtype=np.float64)
        values = pd.to_numeric(df[value_col], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("user_id, song_id, and values must be numeric.") from error

    if (
        not np.isfinite(users).all()
        or not np.isfinite(items).all()
        or (users % 1 != 0).any()
        or (items % 1 != 0).any()
    ):
        raise ValueError("user_id and song_id must contain finite integers.")
    if (users < 0).any() or (users >= n_users).any():
        raise ValueError("user_id values fall outside the requested matrix shape.")
    if (items < 0).any() or (items >= n_items).any():
        raise ValueError("song_id values fall outside the requested matrix shape.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Matrix values must be finite and non-negative.")

    matrix = sparse.csr_matrix(
        (values, (users.astype(np.int64), items.astype(np.int64))),
        shape=(n_users, n_items),
        dtype=np.float64,
    )
    matrix.eliminate_zeros()
    return matrix


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def precision_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Precision@K = |recs ∩ relevant| / K"""
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Recall@K = |recs ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    """F1@K = 2 * P * R / (P + R)"""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = DCG với thứ tự lý tưởng (tất cả relevant đứng đầu)
    """
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # IDCG: giả sử binary relevance
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: np.ndarray, relevant: set, k: int) -> float:
    """
    MRR = 1 / rank_of_first_relevant_item
    Chỉ xét top-K.
    """
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Hit Rate@K = 1 nếu có ít nhất 1 item relevant trong top-K."""
    return float(any(item in relevant for item in recommended[:k]))


def intra_list_diversity_at_k(
    recommended: np.ndarray,
    item_features: sparse.spmatrix,
    k: int,
) -> float:
    """Average pairwise cosine distance between recommended item features."""
    if k < 1:
        raise ValueError("k phải lớn hơn 0.")
    items = np.asarray(recommended[:k], dtype=np.int64)
    if items.size < 2:
        return 0.0
    if (items < 0).any() or (items >= item_features.shape[0]).any():
        raise ValueError("recommended chứa song_id ngoài item_features.")
    if np.unique(items).size != items.size:
        raise ValueError("Danh sách gợi ý không được chứa bài trùng.")

    features = item_features[items]
    similarities = (features @ features.T).toarray()
    upper = similarities[np.triu_indices(items.size, k=1)]
    distances = 1.0 - np.clip(upper, -1.0, 1.0)
    return float(np.mean(distances))


def long_tail_share_at_k(
    recommended: np.ndarray,
    item_popularity: np.ndarray,
    k: int,
    head_fraction: float = 0.2,
) -> float:
    """Share of top-K recommendations outside the popularity head.

    The head is the ``head_fraction`` most interacted-with items, measured
    using training interactions only. Unseen catalog items are in the tail.
    """
    if k < 1:
        raise ValueError("k phải lớn hơn 0.")
    if not 0 < head_fraction < 1:
        raise ValueError("head_fraction phải nằm giữa 0 và 1.")
    popularity = np.asarray(item_popularity, dtype=np.float64)
    if popularity.ndim != 1 or not np.isfinite(popularity).all() or (popularity < 0).any():
        raise ValueError("item_popularity phải là vector hữu hạn, không âm.")

    items = np.asarray(recommended[:k], dtype=np.int64)
    if items.size == 0:
        return 0.0
    if (items < 0).any() or (items >= popularity.size).any():
        raise ValueError("recommended chứa song_id ngoài item_popularity.")
    if np.unique(items).size != items.size:
        raise ValueError("Danh sách gợi ý không được chứa bài trùng.")

    head_size = min(
        popularity.size,
        max(1, int(np.ceil(popularity.size * head_fraction))),
    )
    head_items = np.argsort(-popularity, kind="stable")[:head_size]
    return float(np.mean(~np.isin(items, head_items)))


def evaluate_beyond_accuracy(
    recommend_fn: Callable[[int, int, np.ndarray], np.ndarray],
    train_R: sparse.spmatrix,
    user_ids: np.ndarray,
    item_features: sparse.spmatrix,
    item_popularity: np.ndarray,
    top_k: int = 10,
    head_fraction: float = 0.2,
) -> dict[str, float]:
    """Average ILD@K and LongTailShare@K for a set of users."""
    users = np.asarray(user_ids, dtype=np.int64)
    if users.ndim != 1 or users.size == 0:
        raise ValueError("user_ids phải chứa ít nhất một user.")
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn 0.")
    if train_R.shape[1] != item_features.shape[0]:
        raise ValueError("train_R và item_features không khớp số lượng bài.")
    if train_R.shape[1] != np.asarray(item_popularity).size:
        raise ValueError("train_R và item_popularity không khớp số lượng bài.")
    if (users < 0).any() or (users >= train_R.shape[0]).any():
        raise ValueError("user_ids nằm ngoài train_R.")

    diversity_scores = []
    long_tail_scores = []
    for user_id in users:
        seen = train_R.getrow(int(user_id)).indices
        recommended = np.asarray(
            recommend_fn(int(user_id), top_k, seen), dtype=np.int64
        )
        diversity_scores.append(
            intra_list_diversity_at_k(recommended, item_features, top_k)
        )
        long_tail_scores.append(
            long_tail_share_at_k(
                recommended, item_popularity, top_k, head_fraction=head_fraction
            )
        )
    return {
        "ild": float(np.mean(diversity_scores)),
        "long_tail_share": float(np.mean(long_tail_scores)),
    }


# ─────────────────────────────────────────────
# Evaluator tổng hợp
# ─────────────────────────────────────────────

def evaluate(
    recommend_fn: Callable[[int, int, np.ndarray], np.ndarray],
    train_R: np.ndarray,
    test_df: pd.DataFrame,
    top_k: int = 10,
    relevance_column: str | None = None,
    relevance_threshold: float | None = None,
) -> dict:
    """
    Đánh giá model trên tập test.

    Parameters
    ----------
    recommend_fn : Callable
        Hàm nhận (user_id, top_k, seen_items) → np.ndarray of song_ids.
    train_R : np.ndarray
        Ma trận train (để mask seen items khi recommend).
    test_df : pd.DataFrame
        Ground truth interactions của test set.
    top_k : int
        Số lượng gợi ý.
    relevance_column : str, optional
        Cột dùng xác định bài liên quan, ví dụ rating.
    relevance_threshold : float, optional
        Chỉ coi các tương tác đạt ngưỡng này là relevant.

    Returns
    -------
    dict : Các metric trung bình trên tất cả users trong test.
    """
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn 0.")
    if relevance_threshold is not None and relevance_column is None:
        raise ValueError("Cần relevance_column khi đặt relevance_threshold.")
    if relevance_column is not None:
        if relevance_column not in test_df.columns:
            raise ValueError(f"test_df cần có cột {relevance_column!r}.")
        if relevance_threshold is not None:
            test_df = test_df.loc[test_df[relevance_column] >= relevance_threshold]
    if test_df.empty:
        raise ValueError("Không có tương tác test phù hợp để đánh giá.")

    # Build ground truth: {user_id → set of relevant song_ids}
    ground_truth = (
        test_df.groupby("user_id")["song_id"]
        .apply(set)
        .to_dict()
    )

    metrics = {
        "precision": [], "recall": [], "f1": [],
        "ndcg": [], "mrr": [], "hit_rate": []
    }

    for user_id, relevant_items in ground_truth.items():
        if sparse.issparse(train_R):
            row = train_R.getrow(int(user_id))
            seen = row.indices[row.data > 0]
        else:
            seen = np.where(train_R[user_id] > 0)[0]
        recs = recommend_fn(user_id, top_k, seen)

        p = precision_at_k(recs, relevant_items, top_k)
        r = recall_at_k(recs, relevant_items, top_k)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1_at_k(p, r))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant_items, top_k))
        metrics["mrr"].append(mrr(recs, relevant_items, top_k))
        metrics["hit_rate"].append(hit_rate_at_k(recs, relevant_items, top_k))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def print_metrics(metrics: dict, model_name: str = "Funk SVD", k: int = 10):
    """In bang metric."""
    print("\n" + "-"*50)
    print("  [{}] Evaluation @K={}".format(model_name, k))
    print("-"*50)
    print("  Precision@{:<3} = {:.4f}".format(k, metrics['precision']))
    print("  Recall@{:<5}    = {:.4f}".format(k, metrics['recall']))
    print("  F1@{:<7}        = {:.4f}".format(k, metrics['f1']))
    print("  NDCG@{:<5}      = {:.4f}".format(k, metrics['ndcg']))
    print("  MRR             = {:.4f}".format(metrics['mrr']))
    print("  Hit Rate@{:<3}  = {:.4f}".format(k, metrics['hit_rate']))
    print("-"*50 + "\n")
