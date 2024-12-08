# **Kịch bản Demo**
## **1. Demo app train và đánh giá mô hình XGBoost:**
- Chạy ứng dụng trong folder src/xgboost-package: streamlit app.py.
- Bên phần chọn chức năng nhấn vào nút "Browse files" và chọn tệp "final_data.csv" ở thư mục "data/processed".
- Sau khi đã load được file nhấn vào nút "Huấn luyện" để train mô hình.
- Khi đã huấn luyện thành công ở phần chọn chức năng chọn vào "Đánh giá mô hình".
- Nhấn vào nút "Đánh giá" để đánh giá mô hình với 4 chỉ số MAE, MSE, RMSE, R^2 Score.
- Sau khi đánh giá xong ở phần chức năng chọn "Lưu mô hình" để lưu mô hình đã train vào thư mục models.
## **2. Demo app dự đoán điểm trung bình:**
- Chạy ứng dụng trong folder app: streamlit app.py.
- Nhập các thông tin cần thiết để dự đoán điểm trung bình ở phần giao diện bên tay trái màn hình.
- Sau khi đã nhập xong thông tin nhấn vào nút "Dự đoán" để mô hình dự đoán điểm trung bình học kỳ.
- Sau đó demo với dự liệu thực tế từ dataset để đánh giá kết quả của mô hình.
