"""Block bootstrap of weighted means on small synthetic data."""

import polars as pl
import pytest

from crypto_microstructure.stats import block_bootstrap_wmean


@pytest.fixture
def constant_per_block() -> pl.DataFrame:
    # Group "a": every block has weighted mean 2.0, with different values and weights:
    #   block 0: x 1, 3   w 1, 1  -> (1 + 3) / 2         = 2
    #   block 1: x 2      w 5     -> 10 / 5              = 2
    #   block 2: x 0, 4   w 2, 2  -> (0 + 8) / 4         = 2
    # Group "b": one row per block, x = -1 everywhere, w 1, 1, 2.
    # So any resample of blocks gives 2.0 for "a" and -1.0 for "b".
    # "all" point: sum w*x = (4 + 10 + 8) + (-1 - 1 - 2) = 18,
    #              sum w   = (1 + 1 + 5 + 2 + 2) + (1 + 1 + 2) = 11 + 4 = 15  ->  1.2
    return pl.DataFrame(
        {
            "block": [0, 0, 1, 2, 2, 0, 1, 2],
            "group": ["a", "a", "a", "a", "a", "b", "b", "b"],
            "x": [1.0, 3.0, 2.0, 0.0, 4.0, -1.0, -1.0, -1.0],
            "w": [1.0, 1.0, 5.0, 2.0, 2.0, 1.0, 1.0, 2.0],
        }
    )


def test_constant_per_block_every_resample_equals_point(constant_per_block):
    out = block_bootstrap_wmean(
        constant_per_block, ["x"], "w", "block", group_col="group", n_boot=200, seed=1
    )
    by_group = {r["group"]: r for r in out.iter_rows(named=True)}
    for g, expected in [("a", 2.0), ("b", -1.0)]:
        r = by_group[g]
        assert r["point"] == pytest.approx(expected)
        assert r["ci_lo"] == pytest.approx(expected)
        assert r["ci_hi"] == pytest.approx(expected)
    assert by_group["a"]["n"] == 5 and by_group["a"]["weight"] == pytest.approx(11.0)
    assert by_group["all"]["point"] == pytest.approx(1.2)
    # "all" is not constant per block: block means (4-1)/3 = 1.0, (10-1)/6 = 1.5,
    # (8-2)/6 = 1.0, so resamples differ and the CI has width.
    assert by_group["all"]["ci_lo"] < 1.2 < by_group["all"]["ci_hi"]
