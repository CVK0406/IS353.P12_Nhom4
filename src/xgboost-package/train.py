import os
import pandas as pd
import argparse
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from model import define_model

def preprocess_data(df):
    df.columns = df.columns.str.strip()  
    ordinal_features = ['khoa', 'hedt', 'chuyennganh2']
    df[ordinal_features] = df[ordinal_features].fillna('Unknown')
    
    encoder = OrdinalEncoder()
    df_encoded = encoder.fit_transform(df[ordinal_features])
    df_encoded = pd.DataFrame(
        df_encoded, columns=[f"{col}_mahoa" for col in ordinal_features], index=df.index
    )
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=ordinal_features, inplace=True)

    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    if 'dtbhk' not in df.columns:
        raise ValueError("Dữ liệu đầu vào không chứa cột 'dtbhk' (đầu ra cần dự đoán).")

    X = df.drop(
        columns=['nhom', 'dtbhk', 'id', 'namsinh', 'mssv', 'noisinh', 'diachi_tinhtp', 'Column1', 'lopsh'],
        errors='ignore',
    )
    y = df['dtbhk']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = define_model()
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình XGBoost.")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới file CSV chứa dữ liệu đầu vào.")
    parser.add_argument("--output_dir", type=str, required=True, help="Đường dẫn thư mục lưu mô hình.")
    parser.add_argument("--model_name", type=str, default="xgboost_model.pkl", help="Tên file của mô hình (mặc định: xgboost_model.pkl).")

    args = parser.parse_args()

    try:
        # Đọc dữ liệu
        print(f"Đang tải dữ liệu từ: {args.input}")
        df = pd.read_csv(args.input)
        print("Dữ liệu đã được tải thành công!")

        # Tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Dữ liệu đã được tiền xử lý thành công!")

        # Huấn luyện mô hình
        print("Đang huấn luyện mô hình...")
        model = train_model(X_train, y_train)
        print("Huấn luyện mô hình thành công!")

        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(args.output_dir, exist_ok=True)

        # Lưu mô hình vào thư mục được chỉ định dưới dạng .pkl
        output_path = os.path.join(args.output_dir, args.model_name)
        print(f"Đang lưu mô hình vào: {output_path}")
        joblib.dump(model, output_path)  # Lưu mô hình dưới dạng .pkl
        print("Lưu mô hình thành công!")

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    input("Press Enter to exit...")

