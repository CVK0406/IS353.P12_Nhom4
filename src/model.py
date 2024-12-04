from xgboost import XGBRegressor

def define_model():
    """
    Định nghĩa mô hình XGBoost.
    """
    return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
