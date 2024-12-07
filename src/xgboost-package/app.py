import streamlit as st
import pandas as pd
import joblib
from train import preprocess_data, train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt

# Khởi tạo ứng dụng Streamlit
st.title("Ứng dụng train mô hình XGBoost")
st.sidebar.title("Tùy chọn")

# Khởi tạo trạng thái cho mô hình và dữ liệu
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# Sidebar navigation
menu = st.sidebar.radio("Chọn chức năng", ["Huấn luyện mô hình", "Đánh giá mô hình", "Lưu mô hình"])

if menu == "Huấn luyện mô hình":
    st.header("Huấn luyện mô hình")
    file_path = st.sidebar.file_uploader("Tải lên tệp dữ liệu CSV", type=["csv"])

    if file_path is not None:
        data = pd.read_csv(file_path)
        st.write("Dữ liệu đã tải lên:")
        st.write(data.head())

        if st.button("Huấn luyện"):
            try:
                X_train, X_test, y_train, y_test = preprocess_data(data)
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test
                st.session_state.model = train_model(X_train, y_train)
                st.session_state.trained = True
                st.success("Huấn luyện thành công!")
            except Exception as e:
                st.error(f"Lỗi khi huấn luyện mô hình: {e}")
    else:
        st.warning("Vui lòng tải lên tệp dữ liệu CSV.")

elif menu == "Đánh giá mô hình":
    st.header("Đánh giá mô hình")
    if not st.session_state.trained:
        st.warning("Bạn cần huấn luyện mô hình trước.")
    else:
        if st.button("Đánh giá"):
            try:
                mae, mse, rmse, r2 = evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
                st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
                st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")
            except Exception as e:
                st.error(f"Lỗi khi đánh giá mô hình: {e}")

elif menu == "Lưu mô hình":
    st.header("Lưu mô hình")
    
    if not st.session_state.trained:
        st.warning("Không có mô hình để lưu.")
    else:
        if st.button("Lưu mô hình"):
            try:
                joblib.dump(st.session_state.model, "./models/xgboost_model.pkl")
                st.success("Mô hình đã được lưu thành công dưới tên 'xgboost_model.pkl'.")
            except Exception as e:
                st.error(f"Lỗi khi lưu mô hình: {e}")
