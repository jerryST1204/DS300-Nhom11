"""Rating-aware popularity recommendations from listening history."""

from numbers import Integral

import numpy as np
import pandas as pd


class PopularityBasedRecommendation:
    """Rank tracks by play volume and a Bayesian-smoothed average rating.

    The user ID accepted by ``recommend`` is only for compatibility with the
    shared evaluator; popularity itself is the same for every user.
    """

    def __init__(self, prior_strength: float = 10.0):
        if not np.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and non-negative.")
        self.prior_strength = float(prior_strength)
        self.scores_: np.ndarray | None = None

    def fit(
        self, interactions: pd.DataFrame, n_items: int | None = None
    ) -> "PopularityBasedRecommendation":
        required = {"song_id", "play_count"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(
                f"interactions is missing columns: {', '.join(sorted(missing))}"
            )
        if interactions.empty:
            raise ValueError("interactions must contain at least one row.")

        item_ids = pd.to_numeric(
            interactions["song_id"], errors="raise"
        ).to_numpy(dtype=np.float64)
        play_counts = pd.to_numeric(
            interactions["play_count"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(item_ids).all()
            or (item_ids < 0).any()
            or (item_ids % 1 != 0).any()
        ):
            raise ValueError("song_id must contain non-negative integers.")
        if not np.isfinite(play_counts).all() or (play_counts <= 0).any():
            raise ValueError("play_count must contain finite positive values.")

        if "rating" in interactions:
            ratings = pd.to_numeric(
                interactions["rating"], errors="raise"
            ).to_numpy(dtype=np.float64)
            if (
                not np.isfinite(ratings).all()
                or (ratings < 1).any()
                or (ratings > 5).any()
            ):
                raise ValueError("rating must contain values within [1, 5].")
        else:
            ratings = np.full(len(interactions), 3.0, dtype=np.float64)

        observed_items = item_ids.astype(np.int64)
        inferred_n_items = int(observed_items.max()) + 1
        if n_items is None:
            n_items = inferred_n_items
        if (
            isinstance(n_items, bool)
            or not isinstance(n_items, Integral)
            or n_items < inferred_n_items
        ):
            raise ValueError("n_items must include every observed song_id.")

        total_plays = np.bincount(
            observed_items, weights=play_counts, minlength=int(n_items)
        )
        weighted_rating_sum = np.bincount(
            observed_items, weights=play_counts * ratings, minlength=int(n_items)
        )
        global_mean = float(np.average(ratings, weights=play_counts))
        smoothed_rating = np.divide(
            weighted_rating_sum + self.prior_strength * global_mean,
            total_plays + self.prior_strength,
            out=np.full(int(n_items), global_mean, dtype=np.float64),
            where=(total_plays + self.prior_strength) > 0,
        )

        # A track must have observed plays; its rating quality then adjusts
        # the log-scaled popularity signal.
        self.scores_ = np.zeros(int(n_items), dtype=np.float64)
        observed = total_plays > 0
        self.scores_[observed] = (
            smoothed_rating[observed] * np.log1p(total_plays[observed])
        )
        self.n_items_ = int(n_items)
        return self

    def recommend(
        self,
        user_id: object,
        top_k: int = 10,
        seen_items: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        """Return the highest-ranked unseen items for a user."""
        del user_id
        if self.scores_ is None:
            raise RuntimeError("Call fit(interactions) before recommending.")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, Integral)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        scores = self.scores_.copy()
        if seen_items is not None:
            seen = np.asarray(list(seen_items), dtype=np.int64)
            if (seen < 0).any() or (seen >= self.n_items_).any():
                raise ValueError("seen_items contains an item outside the catalog.")
            scores[seen] = -np.inf
        candidates = np.flatnonzero(scores > 0)
        order = np.argsort(-scores[candidates], kind="stable")
        return candidates[order[: int(top_k)]].astype(np.int64, copy=False)
