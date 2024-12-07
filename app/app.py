import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
import lightgbm as lgb

# Load các mô hình đã huấn luyện
def load_model(model_name):
    if model_name == "XGBoost":
        with open(r"C:\Steam Game\MXH\IS353.P12_Nhom4\app\xgboost_dtbhk_model.pkl", "rb") as file:
            return pickle.load(file)
    elif model_name == "LightGBM":
        with open(r"C:\Steam Game\MXH\IS353.P12_Nhom4\app\light_gbm_dtbhk_model.pkl", "rb") as file:
            return pickle.load(file)

# Hàm chuẩn bị dữ liệu đầu vào
def prepare_input_data(hocky, namhoc, sotchk, gioitinh, khoahoc, tinhtrang, dtbhk2, khoa_mahoa, hedt_mahoa, chuyennganh2_mahoa):
    data = {
        'hocky': [hocky],
        'namhoc': [namhoc],
        'sotchk': [sotchk],
        ' gioitinh': [gioitinh],
        ' khoahoc': [khoahoc],
        ' tinhtrang': [tinhtrang],
        'dtbhk2': [dtbhk2],
        ' khoa_mahoa': [khoa_mahoa],
        ' hedt_mahoa': [hedt_mahoa],
        ' chuyennganh2_mahoa': [chuyennganh2_mahoa],
    }
    return pd.DataFrame(data)

# Giao diện Streamlit
st.title("Dự đoán điểm trung bình học kỳ")

# Chọn mô hình
model_name = st.radio(
    "Chọn mô hình để dự đoán:",
    ("XGBoost", "LightGBM")
)

# Load mô hình đã chọn
model = load_model(model_name)
st.write(f"Đã chọn mô hình: {model_name}")

# Nhập dữ liệu từ người dùng
hocky = st.sidebar.selectbox("Học kỳ ", [1, 2, 3, 4, 6, 7, 8], index=1)
namhoc = st.sidebar.number_input("Năm học", min_value=2000.0, max_value=2030.0, value=2021.0, step=1.0)
sotchk = st.sidebar.number_input("Số tín chỉ", min_value=1.0, max_value=30.0, value=21.0, step=1.0)
gioitinh = st.sidebar.selectbox("Giới tính (0: Nữ, 1: Nam)", [0, 1], index=1)
khoahoc = st.sidebar.number_input("Khóa học", min_value=1.0, max_value=20.0, value=11.0, step=1.0)
tinhtrang = st.sidebar.selectbox("Tình trạng", [0, 1], index=1)
dtbhk2 = st.sidebar.number_input("Điểm trung bình học kỳ trước", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
khoa_mahoa = st.sidebar.selectbox("Khoa (1: HTTT, 2: HTTT, 3: HTTT, 4: HTTT, 5: HTTT)", [1, 2, 3, 4, 5], index=1)
hedt_mahoa = st.sidebar.selectbox("Hệ đào tạo (1: HTTT, 2: HTTT, 3: HTTT, 4: HTTT, 5: HTTT)", [1, 2, 3, 4, 5], index=1)
chuyennganh2_mahoa = st.sidebar.selectbox("Chuyên ngành 2 (1: HTTT, 2: HTTT, 3: HTTT, 4: HTTT, 5: HTTT)", [1, 2, 3, 4, 5], index=1)

# Chuẩn bị dữ liệu
input_data = prepare_input_data(hocky, namhoc, sotchk, gioitinh, khoahoc, tinhtrang, dtbhk2, khoa_mahoa, hedt_mahoa, chuyennganh2_mahoa)

# Nút dự đoán
if st.button("Dự đoán"):
    prediction = model.predict(input_data)
    st.write(f"Dự đoán điểm trung bình học kỳ: {prediction[0]}")

# Load dữ liệu từ CSV
#@st.cache
#def load_data(file_path):
#    return pd.read_csv(file_path)

# Hàm hiển thị thông tin của sinh viên được chọn
#def display_student_info(selected_row):
#    st.text_input("Học kỳ:", value=selected_row["hocky"], key="hocky")
#    st.text_input("Năm học:", value=selected_row["namhoc"], key="namhoc")
#    st.text_input("Số tín chỉ KH:", value=selected_row["sotchk"], key="sotchk")
#    st.text_input("Giới tính:", value=selected_row[" gioitinh"], key=" gioitinh")
#    st.text_input("Khóa học:", value=selected_row[" khoahoc"], key=" khoahoc")
#    st.text_input("Tình trạng:", value=selected_row[" tinhtrang"], key=" tinhtrang")
#    st.text_input("Điểm TB học kỳ 2:", value=selected_row["dtbhk2"], key="dtbhk2")
#    st.text_input("Khoa mã hóa:", value=selected_row[" khoa_mahoa"], key=" khoa_mahoa")
#   st.text_input("Hệ đào tạo mã hóa:", value=selected_row[" hedt_mahoa"], key=" hedt_mahoa")
#    st.text_input("Chuyên ngành 2 mã hóa:", value=selected_row[" chuyennganh2_mahoa"], key=" chuyennganh2_mahoa")

# Đường dẫn đến file CSV của bạn
#file_path = r"C:\Steam Game\MXH\IS353.P12_Nhom4\data\processed\final_data.csv"

# Hiển thị giao diện
#st.title("Danh sách sinh viên")
#data = load_data(file_path)

# Hiển thị bảng danh sách sinh viên
#st.subheader("Danh sách sinh viên:")
#selected_index = st.selectbox(
#    "Chọn sinh viên để xem chi tiết:",
#    options=range(len(data)),
#    format_func=lambda x: f"ID: {data.iloc[x]['id']} - {data.iloc[x]['mssv']}",
#)

# Hiển thị thông tin chi tiết sinh viên
#st.subheader("Thông tin sinh viên:")
#selected_row = data.iloc[selected_index]
#display_student_info(selected_row)
