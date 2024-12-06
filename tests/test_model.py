import pytest
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def define_model():
    """
    Định nghĩa mô hình XGBoost.
    """
    return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

def test_define_model():
    model = define_model()
    assert model is not None  # Kiểm tra mô hình không None
    assert model.n_estimators == 100  # Đảm bảo tham số đúng
    assert model.learning_rate == 0.1
    assert model.max_depth == 3

def test_model_training():
    # Tạo dữ liệu giả lập
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = define_model()
    model.fit(X_train, y_train)  # Huấn luyện mô hình

    # Dự đoán và kiểm tra
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)  # Đảm bảo số lượng dự đoán đúng
