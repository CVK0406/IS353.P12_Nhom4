import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train import preprocess_data # Reuse preprocess_data for consistency


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

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình XGBoost.")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn tới mô hình đã lưu.")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới file CSV chứa dữ liệu đầu vào.")

    args = parser.parse_args()

    try:
        # Kiểm tra file mô hình
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Mô hình không tồn tại tại đường dẫn: {args.model}")

        # Kiểm tra file dữ liệu
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"File dữ liệu không tồn tại tại đường dẫn: {args.input}")

        # Tải mô hình
        print("Đang tải mô hình...")
        model = joblib.load(args.model)
        print("Mô hình đã được tải thành công!")

        # Đọc dữ liệu đầu vào
        print(f"Đang tải dữ liệu từ: {args.input}")
        df = pd.read_csv(args.input)
        print("Dữ liệu đã được tải thành công!")

        # Tiền xử lý dữ liệu
        _, X_test, _, y_test = preprocess_data(df)
        print("Dữ liệu đã được tiền xử lý thành công!")

        # Đánh giá mô hình
        print("Đang đánh giá mô hình...")
        mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Hiển thị kết quả
        print("\nKết quả đánh giá:")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {r2}")

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Nhấn Enter để thoát...")
        input()
    print("Chương trình đã chạy xong. Nhấn Enter để thoát...")
    input()  

