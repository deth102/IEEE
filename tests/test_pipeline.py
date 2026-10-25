"""Unit tests: leak-free protocol, feature definitions and C/Python equivalence.

run:  python -m pytest tests -q      (from D:\\IEEE\\03_code)
"""
import numpy as np
import pytest

from fdedge.config import load_config
from fdedge.data import FeatureStore, blocked_ranges, blocked_split
from fdedge.features import FEATURE_NAMES, extract_features, hamming


@pytest.fixture(scope="module")
def store():
    return FeatureStore(load_config(), "kaist", 8192)


def test_blocked_ranges_partition():
    for L in (10_000, 1_536_000, 7_680_000):
        for f in range(5):
            tr, va, te = blocked_ranges(L, 5, f)
            segs = sorted(tr + va + te)
            assert segs[0][0] == 0 and segs[-1][1] == L
            assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))      # contiguous, no overlap


def test_no_sample_shared_between_train_and_test(store):
    tr, va, te = blocked_split(store, 2)
    n = store.n
    for split_a, split_b in [(tr, te), (tr, va), (va, te)]:
        for r in np.unique(split_b.rec):
            a = np.sort(split_a.start[split_a.rec == r])
            b = split_b.start[split_b.rec == r]
            for s in b:                                   # [s, s+n) must not intersect any [a, a+n)
                i = np.searchsorted(a, s)
                if i < len(a):
                    assert a[i] >= s + n
                if i > 0:
                    assert a[i - 1] + n <= s


def test_features_against_direct_definition():
    rng = np.random.default_rng(1)
    fs, n = 25600.0, 8192
    t = np.arange(n) / fs
    x = (np.sin(2 * np.pi * 50.2 * t) + 0.3 * rng.normal(size=n)).astype(np.float32)
    f = extract_features(x[None], fs)[0]
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isclose(f[1], np.sqrt(np.mean(x.astype(np.float64) ** 2)), rtol=1e-5)
    Y = np.abs(np.fft.fft(x * hamming(n)))[: n // 2]
    k = np.arange(n // 2)
    assert np.isclose(f[13], (k * Y).sum() / Y.sum(), rtol=1e-5)          # spectral centroid (bins)
    assert np.isclose(f[11], f[13] * fs / n, rtol=1e-5)                   # fc = sc * fs / N
    assert abs(np.argmax(Y[9:23]) + 9 - 50.2 * n / fs) <= 1               # 1X found in the band
    assert np.isclose(f[16], Y[9:23].max(), rtol=1e-5)


def test_c_matches_python():
    from tools.build_host import build_and_run
    cfg = load_config()
    if not (cfg.path("firmware_models_dir") / "kaist_n8192_golden" / "fd_testvec.h").exists():
        pytest.skip("run experiments.e12_export first")
    res = build_and_run("kaist_n8192_golden")
    assert res["rf_agree"] == res["n"] and res["rf_q16_agree"] == res["n"] and res["softmax_agree"] == res["n"]
    assert max(res["max_rel_err"]) < 1e-3
