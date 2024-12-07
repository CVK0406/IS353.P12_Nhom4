import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from model import define_model

def preprocess_data(file_path):
    """
    Tiền xử lý dữ liệu từ tệp CSV.
    """
    df = pd.read_csv(r'C:\Steam Game\MXH\IS353.P12_Nhom4\data\processed\final_data.csv')

    
    # Xử lý dữ liệu
    ordinal_features = [' khoa', ' hedt', ' chuyennganh2']
    df[ordinal_features] = df[ordinal_features].fillna('Unknown')
    encoder = OrdinalEncoder()
    df_encoded = encoder.fit_transform(df[ordinal_features])
    df_encoded = pd.DataFrame(df_encoded, columns=[f"{col}_mahoa" for col in ordinal_features], index=df.index)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=ordinal_features, inplace=True)

    # Xử lý NaN
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    # Chọn biến đầu vào và đầu ra
    X = df.drop(columns=['nhom', 'dtbhk', 'id', ' namsinh', 'mssv', ' noisinh', ' diachi_tinhtp', 'Column1', ' lopsh'], errors='ignore')
    y = df['dtbhk']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Huấn luyện mô hình với dữ liệu.
    """
    model = define_model()
    model.fit(X_train, y_train)
    return model
