import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình và hiển thị kết quả.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    return mae, mse, rmse, r2

def plot_feature_importance(model, feature_names):
    """
    Vẽ biểu đồ quan trọng của các đặc trưng và hiển thị trên Streamlit.
    """
    try:
        # Kiểm tra xem mô hình có thuộc tính feature_importances_ không
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            st.error("Mô hình không hỗ trợ feature_importances_")
            return

        # Tạo biểu đồ
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances, color='skyblue')
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance in XGBoost")

        # Hiển thị biểu đồ với Streamlit
        st.pyplot(plt)
    
    except Exception as e:
        st.error(f"Lỗi khi hiển thị biểu đồ Feature Importance: {e}")

