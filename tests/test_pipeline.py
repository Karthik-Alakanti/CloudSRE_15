import numpy as np
import pandas as pd
import pytest
from src.data.pipeline import dirichlet_partition, split_client_indices, normalize_labels, build_preprocessor

def test_dirichlet_partition_conserves_every_index():
    y = np.repeat(np.arange(4), 100)
    parts = dirichlet_partition(y, num_clients=10, alpha=0.3, seed=7)
    flat = np.concatenate(parts)
    assert len(flat) == len(y)
    assert sorted(flat.tolist()) == list(range(len(y)))
    assert all(len(p) > 0 for p in parts)

def test_dirichlet_partition_is_reproducible():
    y = np.repeat(np.arange(3), 50)
    a = dirichlet_partition(y, 5, 0.3, 123)
    b = dirichlet_partition(y, 5, 0.3, 123)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))

def test_client_splits_are_disjoint_and_conserve_indices():
    y = np.repeat(np.arange(3), 30)
    idx = np.arange(len(y))
    train, val, cal = split_client_indices(idx, y, seed=9)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(cal)) == 0
    assert len(set(val) & set(cal)) == 0
    assert sorted(np.concatenate([train, val, cal]).tolist()) == idx.tolist()

def test_normalize_labels_rejects_missing():
    with pytest.raises(ValueError):
        normalize_labels(pd.Series(["Normal", None]))

def test_preprocessor_excludes_target_and_identity_columns():
    df = pd.DataFrame({
        "id": [1,2,3,4],
        "attack_cat": ["Normal","DoS","Normal","DoS"],
        "label": [0,1,0,1],
        "srcip": ["a","b","c","d"],
        "dur": [0.1,0.2,0.3,0.4],
        "proto": ["tcp","udp","tcp","udp"],
    })
    _, X, meta = build_preprocessor(df)
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns
    assert "id" not in X.columns
    assert "srcip" not in X.columns
    assert set(X.columns) == {"dur","proto"}


def test_preprocessor_excludes_ip_and_time_columns_case_insensitive():
    df = pd.DataFrame({
        "id": [1,2,3,4],
        "attack_cat": ["Normal","DoS","Normal","DoS"],
        "label": [0,1,0,1],
        "srcip": ["a","b","c","d"],
        "dstip": ["e","f","g","h"],
        "Stime": [1.0,2.0,3.0,4.0],
        "Ltime": [5.0,6.0,7.0,8.0],
        "dur": [0.1,0.2,0.3,0.4],
        "proto": ["tcp","udp","tcp","udp"],
    })
    _, X, meta = build_preprocessor(df)
    for forbidden in ("id","attack_cat","label","srcip","dstip","stime","ltime"):
        assert forbidden not in {str(c).lower() for c in X.columns}, forbidden
    assert set(str(c).lower() for c in X.columns) == {"dur","proto"}


def test_preprocessor_handles_infinity_without_non_finite_output():
    # The pipeline replaces +/-inf with NaN before fitting (pipeline.py:146);
    # mirror that here, then assert the transformed output is finite.
    df = pd.DataFrame({
        "id": [1,2,3,4],
        "attack_cat": ["Normal","DoS","Normal","DoS"],
        "label": [0,1,0,1],
        "dur": [0.1, float("inf"), 0.3, float("-inf")],
        "proto": ["tcp","udp","tcp","udp"],
    })
    transformer, X, meta = build_preprocessor(df)
    X = X.replace([np.inf, -np.inf], np.nan)
    Xt = transformer.fit_transform(X)
    assert np.isfinite(Xt.data).all()


def test_dirichlet_partition_rare_class_allows_class_absence():
    # One rare class with a single sample: only one client can hold it; all
    # indices must still be conserved exactly once across clients.
    y = np.array([0]*40 + [1])  # class 1 is a singleton
    parts = dirichlet_partition(y, num_clients=10, alpha=0.3, seed=5, min_size=0)
    flat = np.concatenate(parts)
    assert len(flat) == len(y)
    assert sorted(flat.tolist()) == list(range(len(y)))
    owners = [int(np.any(p == len(y)-1)) for p in parts]
    assert sum(owners) == 1  # singleton class lands in exactly one client


def test_client_split_stratified_preserves_every_class_in_train():
    # Each locally present class must have at least one training sample.
    indices = np.arange(30)
    y = np.repeat(np.arange(3), 10)
    train, val, cal = split_client_indices(indices, y, seed=1)
    assert set(np.unique(y[train])) == set(np.unique(y))
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(cal)) == 0
    assert len(set(val) & set(cal)) == 0
