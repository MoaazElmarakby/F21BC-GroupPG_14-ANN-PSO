import numpy as np
import pytest

from psoann.config import DataConfig
from psoann.data import Standardizer, load_csv, make_split, stratified_kfold, stratified_split


def test_banknote_loads_with_the_expected_shape():
    X, y = load_csv("data/raw/banknote.csv")
    assert X.shape == (1372, 4) and y.shape == (1372,)
    assert np.bincount(y).tolist() == [762, 610]


def test_loader_rejects_bad_labels(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("1.0,2.0,0\n1.0,2.0,2\n")
    with pytest.raises(ValueError, match="contiguous"):
        load_csv(bad)
    frac = tmp_path / "frac.csv"
    frac.write_text("1.0,2.0,0.5\n")
    with pytest.raises(ValueError, match="non-negative integers"):
        load_csv(frac)
    with pytest.raises(FileNotFoundError):
        load_csv("does/not/exist.csv")


def test_stratified_split_partitions_and_keeps_proportions(rng):
    y = np.array([0] * 80 + [1] * 20)
    train, test = stratified_split(y, 0.7, rng)
    assert len(set(train) & set(test)) == 0 and len(train) + len(test) == 100
    assert np.bincount(y[train]).tolist() == [56, 14] and np.bincount(y[test]).tolist() == [24, 6]


def test_split_is_deterministic_given_the_generator():
    y = np.array([0] * 30 + [1] * 30)
    a = stratified_split(y, 0.5, np.random.default_rng(5))
    b = stratified_split(y, 0.5, np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], stratified_split(y, 0.5, np.random.default_rng(6))[0])


def test_kfold_covers_every_sample_once_in_test(rng):
    y = np.array([0] * 40 + [1] * 25)
    folds = stratified_kfold(y, 5, rng)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests.tolist()) == list(range(65))
    for train, test in folds:
        assert not set(train) & set(test) and len(train) + len(test) == 65


def test_standardiser_uses_training_statistics_only(rng):
    X = rng.normal(5.0, 3.0, size=(100, 3))
    y = np.array([0, 1] * 50)
    split = make_split(X, y, DataConfig(train_fraction=0.7), rng)
    np.testing.assert_allclose(split.X_train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(split.X_train.std(axis=0), 1.0, atol=1e-9)
    assert not np.allclose(split.X_test.mean(axis=0), 0.0, atol=1e-9)  # test set is NOT re-centred
    raw = make_split(X, y, DataConfig(train_fraction=0.7, standardise=False), np.random.default_rng(1234))
    np.testing.assert_array_equal(raw.X_train, X[raw.train_index])


def test_constant_feature_does_not_produce_nan():
    out = Standardizer().fit(np.array([[1.0, 2.0], [1.0, 4.0]])).transform(np.array([[1.0, 3.0]]))
    assert np.all(np.isfinite(out)) and out[0, 0] == 0.0
