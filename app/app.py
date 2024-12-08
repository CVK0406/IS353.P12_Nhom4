import os
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
import lightgbm as lgb

# Load các mô hình đã huấn luyện
def load_model(model_name):
    try:
        if model_name == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
        elif model_name == "LightGBM":
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'light_gbm_model.pkl')
        else:
            st.error("Mô hình không hợp lệ!")
            return None

        # Kiểm tra nếu tệp không tồn tại
        if not os.path.exists(model_path):
            st.error(f"Không tìm thấy mô hình: {model_path}")
            return None

        # Tải mô hình
        return joblib.load(model_path)  
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

# Hàm chuẩn bị dữ liệu đầu vào
def prepare_input_data(hocky, namhoc, sotchk, gioitinh, khoahoc, tinhtrang, dtbhk2, khoa_mahoa, hedt_mahoa, chuyennganh2_mahoa):
    try:
        data = {
            'hocky': [hocky],
            'namhoc': [namhoc],
            'sotchk': [sotchk],
            'gioitinh': [gioitinh],
            'khoahoc': [khoahoc],
            'tinhtrang': [tinhtrang],
            'dtbhk2': [dtbhk2],
            'khoa_mahoa': [khoa_mahoa],
            'hedt_mahoa': [hedt_mahoa],
            'chuyennganh2_mahoa': [chuyennganh2_mahoa],  
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Lỗi khi chuẩn bị dữ liệu đầu vào: {e}")
        return None

# Giao diện Streamlit
st.title("Dự đoán điểm trung bình học kỳ")

# Chọn mô hình
model_name = st.radio(
    "Chọn mô hình để dự đoán:",
    ("XGBoost", "LightGBM")
)

# Load mô hình đã chọn
model = load_model(model_name)

if model is not None:
    st.write(f"Đã chọn mô hình: {model_name}")

    # Nhập dữ liệu từ người dùng
    hocky = st.sidebar.selectbox("Học kỳ ", [1, 2, 3, 4, 6, 7, 8], index=1)
    namhoc = st.sidebar.number_input("Năm học", min_value=2000.0, max_value=2030.0, value=2021.0, step=1.0)
    sotchk = st.sidebar.number_input("Số tín chỉ", min_value=1.0, max_value=30.0, value=21.0, step=1.0)
    gioitinh = st.sidebar.selectbox("Giới tính (0: Nữ, 1: Nam)", [0, 1], index=1)
    khoahoc = st.sidebar.number_input("Khóa học", min_value=1.0, max_value=20.0, value=11.0, step=1.0)
    tinhtrang = st.sidebar.selectbox("Tình trạng", [0, 1], index=1)
    dtbhk2 = st.sidebar.number_input("Điểm trung bình học kỳ trước", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    khoa_mahoa = st.sidebar.selectbox("Khoa (0: CNPM, 1: HTTT, 2: KHMT, 3: KTMT, 4: KTTT, 5: MMT&TT)", [0, 1, 2, 3, 4, 5], index=1)
    hedt_mahoa = st.sidebar.selectbox("Hệ đào tạo (0: CLC, 1: CNTN, 2: CQUI, 3: CTTT, 4: KSTN)", [0, 1, 2, 3, 4], index=1)
    chuyennganh2_mahoa = st.sidebar.selectbox("Chuyên ngành 2 (0: 7480201_CLCN, 1: 7480201_KHDL, 2: D480101, 3: D480102, 4: D480103, 5: D480104, 6: D480201, 7: D480299, 8: D520214, 9: D52480104, 10: 7480102, 11: 7480109)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], index=1)

    # Chuẩn bị dữ liệu
    input_data = prepare_input_data(hocky, namhoc, sotchk, gioitinh, khoahoc, tinhtrang, dtbhk2, khoa_mahoa, hedt_mahoa, chuyennganh2_mahoa)

    if input_data is not None:
        # Nút dự đoán
        if st.button("Dự đoán"):
            try:
                prediction = model.predict(input_data)
                st.write(f"Dự đoán điểm trung bình học kỳ: {prediction[0]}")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
else:
    st.warning("Không tải được mô hình. Vui lòng kiểm tra lại.")
