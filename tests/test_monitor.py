import numpy as np
import pandas as pd

from src.monitor import drift_report


def _df(seed, shift=0.0, n=2000):
    rng = np.random.default_rng(seed)
    data = {f"num{i}": rng.normal(shift, 1, n) for i in range(4)}
    data.update({f"cat{i}": rng.choice(["x", "y", "z"], n) for i in range(4)})
    return pd.DataFrame(data)


def test_no_drift_on_same_distribution():
    assert not drift_report(_df(0), _df(1))["dataset_drift"]


def test_drift_detected_on_shift():
    report = drift_report(_df(0), _df(1, shift=1.0))
    assert report["dataset_drift"]
    assert report["columns"]["num0"]["drift"]