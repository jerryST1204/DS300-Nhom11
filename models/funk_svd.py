"""
funk_svd.py
───────────
Thuật toán Funk SVD (Matrix Factorization) sử dụng Stochastic Gradient Descent (SGD)
dựa trên slide Buổi 4 môn Hệ Khuyến Nghị (DS300).

Mô hình:
    r̂_{ui} = μ + b_u + b_i + p_u^T q_i

Hàm mất mát báo cáo (trung bình trên các rating quan sát):
    L = MSE + λ/N * [Σ_u n_u (b_u² + ||p_u||²)
                     + Σ_i n_i (b_i² + ||q_i||²)]

Quy tắc cập nhật SGD:
    e_{ui} = r_{ui} - (μ + b_u + b_i + p_u^T q_i)
    b_u ← b_u + η (e_{ui} - λ b_u)
    b_i ← b_i + η (e_{ui} - λ b_i)
    p_{old} = p_u
    p_u ← p_u + η * (e_{ui} * q_i - λ * p_u)
    q_i ← q_i + η * (e_{ui} * p_{old} - λ * q_i)
"""

import numpy as np
import time
from scipy import sparse
from typing import Optional


class FunkSVD:
    """
    Funk SVD Recommender sử dụng SGD.

    Parameters
    ----------
    factors : int
        Số chiều của không gian ẩn (latent dimension k).
    learning_rate : float
        Tốc độ học (learning rate η).
    reg : float
        Hệ số điều chuẩn (regularization λ).
    n_epochs : int
        Số vòng lặp qua toàn bộ dữ liệu huấn luyện.
    init_std : float
        Độ lệch chuẩn khi khởi tạo trọng số ngẫu nhiên.
    random_state : int
        Seed cho việc sinh số ngẫu nhiên để tái lặp kết quả.
    verbose : bool
        In thông tin loss/RMSE sau mỗi epoch.
    """

    def __init__(
        self,
        factors: int = 32,
        learning_rate: float = 0.01,
        reg: float = 0.5,
        n_epochs: int = 4,
        init_std: float = 0.05,
        random_state: int = 42,
        verbose: bool = True,
    ):
        if factors < 1:
            raise ValueError("factors phải lớn hơn 0.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate phải là số hữu hạn lớn hơn 0.")
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg phải là số hữu hạn không âm.")
        if n_epochs < 1:
            raise ValueError("n_epochs phải lớn hơn 0.")
        if not np.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std phải là số hữu hạn lớn hơn 0.")

        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.init_std = init_std
        self.random_state = random_state
        self.verbose = verbose

        self.n_users: int = 0
        self.n_items: int = 0
        self.global_mean: float = 0.0
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None  # P: (n_users, factors)
        self.item_factors: Optional[np.ndarray] = None  # Q: (n_items, factors)
        self.history: list[dict] = []

    def fit(self, R: np.ndarray) -> "FunkSVD":
        """
        Huấn luyện mô hình Funk SVD trên ma trận tương tác/rating R.
        Chỉ tính toán và cập nhật trên các ô đã có tương tác (R[u, i] > 0).

        Parameters
        ----------
        R : np.ndarray, shape (n_users, n_items)
            Ma trận rating / lượt nghe (0 biểu thị chưa quan sát/chưa nghe).
        """
        if sparse.issparse(R):
            matrix = sparse.csr_matrix(R, dtype=np.float64, copy=True)
            if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
                raise ValueError("R phải là ma trận 2 chiều có ít nhất 1 user và 1 item.")
            if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
                raise ValueError("R chỉ được chứa các giá trị hữu hạn, không âm.")
            matrix.eliminate_zeros()
            observations = matrix.tocoo(copy=False)
            users, items, ratings = observations.row, observations.col, observations.data
            matrix_shape = matrix.shape
        else:
            R = np.asarray(R, dtype=np.float64)
            if R.ndim != 2 or not R.shape[0] or not R.shape[1]:
                raise ValueError("R phải là ma trận 2 chiều có ít nhất 1 user và 1 item.")
            if not np.isfinite(R).all() or np.any(R < 0):
                raise ValueError("R chỉ được chứa các giá trị hữu hạn, không âm.")
            users, items = np.nonzero(R)
            ratings = R[users, items]
            matrix_shape = R.shape

        if not matrix_shape[0] or not matrix_shape[1]:
            raise ValueError("R phải là ma trận 2 chiều có ít nhất 1 user và 1 item.")

        n_ratings = len(ratings)
        if n_ratings == 0:
            raise ValueError("R phải có ít nhất một tương tác dương để huấn luyện.")

        self.n_users, self.n_items = matrix_shape
        self.global_mean = float(np.mean(ratings))
        user_counts = np.bincount(users, minlength=self.n_users).astype(np.float64)
        item_counts = np.bincount(items, minlength=self.n_items).astype(np.float64)
        rng = np.random.default_rng(self.random_state)
        self.user_biases = np.zeros(self.n_users, dtype=np.float64)
        self.item_biases = np.zeros(self.n_items, dtype=np.float64)
        self.user_factors = rng.normal(0, self.init_std, size=(self.n_users, self.factors))
        self.item_factors = rng.normal(0, self.init_std, size=(self.n_items, self.factors))
        self.history.clear()

        if self.verbose:
            print(f"[Funk SVD] Training: {self.n_users} users, {self.n_items} items, "
                  f"{n_ratings} ratings, k={self.factors}, lr={self.learning_rate}, reg={self.reg}")

        indices = np.arange(n_ratings)

        for epoch in range(1, self.n_epochs + 1):
            t_epoch_start = time.time()
            # Xáo trộn dữ liệu mỗi epoch cho SGD
            rng.shuffle(indices)

            # Thực hiện cập nhật SGD từng cặp (u, i)
            for idx in indices:
                u = users[idx]
                i = items[idx]
                r_ui = ratings[idx]

                # Dự đoán rating: r̂_{ui} = p_u^T q_i
                pred = (
                    self.global_mean
                    + self.user_biases[u]
                    + self.item_biases[i]
                    + np.dot(self.user_factors[u], self.item_factors[i])
                )
                e_ui = r_ui - pred

                # Cập nhật bias và latent factors với SGD.
                self.user_biases[u] += self.learning_rate * (
                    e_ui - self.reg * self.user_biases[u]
                )
                self.item_biases[i] += self.learning_rate * (
                    e_ui - self.reg * self.item_biases[i]
                )
                p_old = self.user_factors[u].copy()
                grad_u = e_ui * self.item_factors[i] - self.reg * self.user_factors[u]
                grad_i = e_ui * p_old - self.reg * self.item_factors[i]
                self.user_factors[u] += self.learning_rate * grad_u
                self.item_factors[i] += self.learning_rate * grad_i


            # Tính MSE/RMSE và objective trung bình trên rating quan sát.
            preds = (
                self.global_mean
                + self.user_biases[users]
                + self.item_biases[items]
                + np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
            )
            errors = ratings - preds
            mse = float(np.mean(errors ** 2))
            rmse = float(np.sqrt(mse))
            user_norms = np.einsum("ij,ij->i", self.user_factors, self.user_factors)
            item_norms = np.einsum("ij,ij->i", self.item_factors, self.item_factors)
            reg_loss = self.reg * (
                np.dot(user_counts, user_norms + self.user_biases ** 2)
                + np.dot(item_counts, item_norms + self.item_biases ** 2)
            ) / n_ratings
            total_loss = mse + reg_loss

            epoch_time = time.time() - t_epoch_start
            self.history.append({
                "epoch": epoch,
                "mse": mse,
                "rmse": rmse,
                "reg_loss": float(reg_loss),
                "loss": float(total_loss),
                "time": epoch_time,
            })

            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.n_epochs):
                print(f"  Epoch {epoch:2d}/{self.n_epochs:2d} | "
                      f"RMSE: {rmse:.4f} | Mean Loss: {total_loss:.4f} | Time: {epoch_time:.2f}s")

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Dự đoán điểm tương tác / rating của user đối với item."""
        if (
            self.user_factors is None or self.item_factors is None
            or self.user_biases is None or self.item_biases is None
        ):
            raise ValueError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(f"user_id phải nằm trong khoảng 0..{self.n_users - 1}.")
        if item_id < 0 or item_id >= self.n_items:
            raise ValueError(f"item_id phải nằm trong khoảng 0..{self.n_items - 1}.")
        return float(
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases[item_id]
            + np.dot(self.user_factors[user_id], self.item_factors[item_id])
        )

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Gợi ý top-K bài hát cho một user.

        Parameters
        ----------
        user_id : int
            ID của user cần gợi ý.
        top_k : int
            Số lượng bài hát gợi ý.
        exclude_seen : bool
            Nếu True, loại bỏ những bài hát user đã nghe.
        seen_items : np.ndarray, optional
            Danh sách ID bài hát đã nghe.

        Returns
        -------
        np.ndarray: Mảng chứa top_k item_id được xếp hạng cao nhất.
        """
        if (
            self.user_factors is None or self.item_factors is None
            or self.user_biases is None or self.item_biases is None
        ):
            raise ValueError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(f"user_id phải nằm trong khoảng 0..{self.n_users - 1}.")
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0.")

        # Bias/item baseline improves ranking while preserving latent signals.
        scores = (
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases
            + np.dot(self.user_factors[user_id], self.item_factors.T)
        )

        if exclude_seen and seen_items is not None and len(seen_items) > 0:
            scores[seen_items] = -np.inf

        # Lấy top-K bài còn có thể gợi ý; top_k có thể lớn hơn số bài chưa nghe.
        candidates = np.flatnonzero(np.isfinite(scores))
        top_indices = candidates[np.argsort(-scores[candidates])[:top_k]]
        return top_indices
