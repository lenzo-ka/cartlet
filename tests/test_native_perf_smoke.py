"""Native training throughput smoke test.

Guards against a regression to the old O(n^2)-per-feature-per-node numeric
split search. The threshold is intentionally generous: the incremental sweep
trains this dataset in well under a second, while the quadratic version took
tens of seconds, so this only fires on an order-of-magnitude regression (a
reintroduced quadratic scan), not on normal machine-to-machine variance.
"""

from __future__ import annotations

import random
import time

from cartlet import DecisionTree

# Wall-clock ceiling. The incremental search is ~0.2s here; the old quadratic
# search was ~15s. A 10s ceiling clears CI variance by ~50x while still
# catching a return to quadratic behavior.
_CEILING_SECONDS = 10.0


def test_native_numeric_training_is_subquadratic():
    rng = random.Random(0)
    n_rows, n_features = 2500, 12
    X = [[rng.random() for _ in range(n_features)] for _ in range(n_rows)]
    y = [str(rng.randint(0, 2)) for _ in range(n_rows)]
    specs = [
        {"name": f"f{i}", "dtype": "float", "type": "num"} for i in range(n_features)
    ]

    dt = DecisionTree(features=specs)
    dt.load_data(X, y)

    start = time.time()
    dt.train()
    elapsed = time.time() - start

    assert elapsed < _CEILING_SECONDS, (
        f"native numeric training took {elapsed:.1f}s (ceiling {_CEILING_SECONDS}s); "
        "a quadratic split search may have been reintroduced"
    )
    # Sanity: it actually built a tree.
    assert dt.predict(X[0]) in {"0", "1", "2"}
