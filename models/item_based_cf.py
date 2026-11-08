"""Item-Based CF: dùng model.fit / model.recommend với ma trận play_count train.

Hàng là người dùng, cột là bài hát. Ô 0 là chưa có tương tác, không phải
không thích. Thuật toán chỉ dùng lịch sử nghe, không dùng thuộc tính bài hát.
"""

from numbers import Integral

import numpy as np
from scipy import sparse


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} phải là số nguyên >= {minimum}.")
    return int(value)


class ItemBasedCF:
    """Cosine giữa bài hát + K bài đã nghe tương đồng + điểm có trọng số.

    k là số bài láng giềng tối đa cho mỗi bài cần dự đoán, ví dụ 5, 10, 20.
    Điểm là tổng similarity * log1p(play_count), không phải rating 1–5.
    """

    def __init__(self, k=10):
        self.k = _integer(k, "k", minimum=1)
        self._fitted = False

    def fit(self, train_R):
        """Nhận ma trận (n_users, n_songs) chỉ có train và trả lại chính model."""
        try:
            counts = sparse.csr_matrix(train_R, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("train_R phải là ma trận số 2 chiều.") from error
        if counts.ndim != 2 or 0 in counts.shape:
            raise ValueError("train_R phải là ma trận 2 chiều có người dùng và bài hát.")
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError("play_count phải hữu hạn và không âm.")

        # 1. Giảm ảnh hưởng của lượt nghe lớn; giữ nguyên chỉ số hàng/cột.
        strengths = counts.copy()
        strengths.data = np.log1p(strengths.data)

        # 2. Mỗi cột là vector một bài hát trên toàn bộ người dùng trong train.
        lengths = np.sqrt(
            np.asarray(strengths.multiply(strengths).sum(axis=0)).ravel()
        )
        inverse_lengths = np.divide(
            1.0, lengths, out=np.zeros_like(lengths), where=lengths > 0
        )
        normalized = strengths @ sparse.diags(inverse_lengths, format="csr")
        similarity = np.zeros(
            (counts.shape[1], counts.shape[1]), dtype=np.float64
        )
        for user in range(normalized.shape[0]):
            row = normalized.getrow(user)
            items = row.indices
            if len(items) < 2:
                continue
            values = row.data
            similarity[np.ix_(items, items)] += np.outer(values, values)
        similarity = np.clip(similarity, 0.0, 1.0)
        np.fill_diagonal(similarity, 0.0)  # Một bài không là láng giềng của chính nó.

        self.train_matrix_ = counts
        self.interaction_matrix_ = strengths
        self.similarity_ = similarity
        self.n_users_, self.n_items_ = counts.shape
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Cần gọi model.fit(train_R) trước.")

    def _check_user(self, user_id):
        self._check_fitted()
        user = _integer(user_id, "user_id")
        if user >= self.n_users_:
            raise ValueError("user_id nằm ngoài các hàng của train_R.")
        return user

    def _check_song(self, song_id):
        self._check_fitted()
        song = _integer(song_id, "song_id")
        if song >= self.n_items_:
            raise ValueError("song_id nằm ngoài các cột của train_R.")
        return song

    def get_neighbors(self, song_id, user_id=None):
        """Trả tối đa k cặp (chỉ số bài, cosine) có tương đồng dương giảm dần.

        Có user_id: chỉ chọn trong các bài người đó đã nghe ở train; đây là
        tập láng giềng thực sự dùng để tính điểm. Không có user_id: xem k bài
        tương đồng nhất toàn cục để giải thích ma trận Item–Item Cosine.
        """
        song = self._check_song(song_id)
        similarities = self.similarity_[song]
        eligible = similarities > 0
        if user_id is not None:
            user = self._check_user(user_id)
            listened = self.train_matrix_.getrow(user).indices
            eligible &= np.isin(np.arange(self.n_items_), listened)
        candidates = np.flatnonzero(eligible)
        order = np.argsort(-similarities[candidates], kind="stable")
        neighbors = candidates[order[:self.k]]
        return [(int(item), float(similarities[item])) for item in neighbors]

    def predict_all(self, user_id):
        """Điểm từng cột bài hát đúng thứ tự train_R; chưa lọc bài đã nghe."""
        user = self._check_user(user_id)
        scores = np.zeros(self.n_items_, dtype=float)
        train_row = self.train_matrix_.getrow(user)
        listened = train_row.indices[train_row.data > 0]
        if not len(listened):
            return scores

        # Select each target track's top-k neighbors from this user's history
        # in one NumPy operation instead of sorting once per track.
        similarities = self.similarity_[:, listened]
        n_history = len(listened)
        if n_history <= self.k:
            selected = similarities > 0
        else:
            threshold_index = n_history - self.k
            thresholds = np.partition(
                similarities, threshold_index, axis=1
            )[:, threshold_index]
            selected = similarities > thresholds[:, np.newaxis]

            # Keep stable item-id ordering when neighbors tie at the top-k
            # boundary, matching get_neighbors().
            tie_rows = np.flatnonzero(
                (thresholds > 0) & (selected.sum(axis=1) < self.k)
            )
            for row in tie_rows:
                threshold = thresholds[row]
                tied = np.flatnonzero(similarities[row] == threshold)
                needed = self.k - int(selected[row].sum())
                selected[row, tied[:needed]] = True

        weights = np.where(selected, similarities, 0.0)
        # Keep similarity magnitude in the score. A normalized weighted
        # average makes every one-neighbor candidate inherit the exact same
        # score, regardless of how weakly it resembles that neighbor.
        user_strengths = self.interaction_matrix_.getrow(user).toarray().ravel()[listened]
        return weights @ user_strengths

    def predict(self, user_id, song_id):
        """Điểm quan tâm của một người dùng với một cột bài hát."""
        user = self._check_user(user_id)
        song = self._check_song(song_id)
        return float(self.predict_all(user)[song])

    def recommend(self, user_id, top_k=10, seen_items=None):
        """Trả mảng chỉ số bài chưa nghe, tương thích cách gọi trong notebook.

        top_k là số bài gợi ý, khác self.k là số bài láng giềng để tính điểm.
        Luôn loại bài trong train và các bài bổ sung qua seen_items. Chỉ trả
        bài có bằng chứng CF; user hàng 0 trả rỗng, không thêm Popularity.
        """
        user = self._check_user(user_id)
        top_k = _integer(top_k, "top_k", minimum=1)
        scores = self.predict_all(user)
        excluded = np.zeros(self.n_items_, dtype=bool)
        train_row = self.train_matrix_.getrow(user)
        excluded[train_row.indices[train_row.data > 0]] = True
        if seen_items is not None:
            try:
                items = list(seen_items)
            except TypeError as error:
                raise ValueError("seen_items phải là danh sách chỉ số bài hát.") from error
            for value in items:
                song = _integer(value, "seen_items")
                if song >= self.n_items_:
                    raise ValueError("seen_items có chỉ số ngoài các cột của train_R.")
                excluded[song] = True

        # 5. Loại bài đã nghe; hòa điểm thì ưu tiên chỉ số cột nhỏ hơn.
        candidates = np.flatnonzero((scores > 0) & ~excluded)
        order = np.argsort(-scores[candidates], kind="stable")
        return candidates[order[:top_k]].astype(np.int64, copy=False)
