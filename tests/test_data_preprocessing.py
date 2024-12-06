import pytest
import pandas as pd

@pytest.fixture
def load_raw_data():
    df1 = pd.DataFrame({
        'mssv': [1, 2, 3],
        'dtbhk': [7.5, 8.0, 9.0],
        'sotchk': [15, 20, 10]
    })
    df2 = pd.DataFrame({
        'mssv': [1, 2, 3],
        '_1': [0, 0, 0],
        'namhoc': [2020, 2021, 2022],
        'hocky': [1, 2, 1]
    })
    return df1, df2

def test_merge_data(load_raw_data):
    df1, df2 = load_raw_data
    
    # Loại bỏ cột '_1'
    df2 = df2.drop(columns=['_1'])
    assert '_1' not in df2.columns

    # Hợp nhất dữ liệu
    df_merged = pd.merge(df1, df2, on='mssv', how='outer')
    assert df_merged.shape[0] == max(len(df1), len(df2))  # Đảm bảo số hàng đúng

def test_outlier_removal():
    df = pd.DataFrame({
        'sotchk': [10, 15, 20, 1000],  # Giá trị 1000 là outlier
        'dtbhk': [6.0, 7.0, 8.0, 9.0]
    })

    Q1 = df['sotchk'].quantile(0.25)
    Q3 = df['sotchk'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df['sotchk'] >= lower_bound) & (df['sotchk'] <= upper_bound)]
    assert len(df_filtered) == 3  # Dòng chứa outlier bị loại bỏ

def test_shift_column():
    df = pd.DataFrame({
        'mssv': [1, 1, 1],
        'dtbhk': [7.0, 8.0, 9.0]
    })

    df['dtbhk2'] = df.groupby('mssv')['dtbhk'].shift(1)
    assert df['dtbhk2'].isnull().sum() == 1  # Dòng đầu tiên bị NaN
    assert df['dtbhk2'].iloc[1] == 7.0  # Kiểm tra giá trị đã được shift
