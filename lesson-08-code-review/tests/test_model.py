import pytest
import pandas as pd
import numpy as np
import json
from src.model import TaxiFareModel


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X_train = pd.DataFrame({"feature1": np.random.rand(100), "feature2": np.random.rand(100)})
    y_train = pd.Series(np.random.rand(100) * 100)
    X_test = pd.DataFrame({"feature1": np.random.rand(20), "feature2": np.random.rand(20)})
    y_test = pd.Series(np.random.rand(20) * 100)
    return X_train, X_test, y_train, y_test


def test_model_initialization():
    model = TaxiFareModel()
    assert model.model is not None
    assert model.model.random_state == 42


def test_model_fit(sample_data):
    X_train, _, y_train, _ = sample_data
    model = TaxiFareModel()
    model.fit(X_train, y_train)
    assert hasattr(model.model, "estimators_")


def test_model_fit_empty_data():
    model = TaxiFareModel()
    X_empty = pd.DataFrame()
    y_empty = pd.Series()
    with pytest.raises(ValueError, match="empty"):
        model.fit(X_empty, y_empty)


def test_model_fit_mismatched_length():
    model = TaxiFareModel()
    X = pd.DataFrame({"feature1": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError, match="same length"):
        model.fit(X, y)


def test_model_predict(sample_data):
    X_train, X_test, y_train, _ = sample_data
    model = TaxiFareModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)


def test_model_predict_empty_data(sample_data):
    X_train, _, y_train, _ = sample_data
    model = TaxiFareModel()
    model.fit(X_train, y_train)

    with pytest.raises(ValueError, match="cannot be empty"):
        model.predict(pd.DataFrame())


def test_model_evaluate(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = TaxiFareModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    assert "r2_score" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_model_save_and_load(sample_data, tmp_path):
    X_train, X_test, y_train, y_test = sample_data
    model = TaxiFareModel()
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)

    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    model.save(str(model_path), str(metrics_path))

    assert model_path.exists()
    assert metrics_path.exists()

    with open(metrics_path) as f:
        saved_metrics = json.load(f)

    assert "metrics" in saved_metrics
    assert "model_params" in saved_metrics
    assert "n_features" in saved_metrics

    loaded_model = TaxiFareModel.load(str(model_path))
    predictions_original = model.predict(X_test)
    predictions_loaded = loaded_model.predict(X_test)

    np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)


def test_model_reproducibility():
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40, 50])

    model1 = TaxiFareModel(random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict(X)

    model2 = TaxiFareModel(random_state=42)
    model2.fit(X, y)
    pred2 = model2.predict(X)

    np.testing.assert_array_equal(pred1, pred2)
