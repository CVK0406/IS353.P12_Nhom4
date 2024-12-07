import pandas as pd

df1 = pd.read_csv('C:\Steam Game\MXH\IS353.P12_Nhom4\data\raw\sinhvien_dtb_hocky.csv')

df2 = pd.read_csv('C:\Steam Game\MXH\IS353.P12_Nhom4\data\raw\01.sinhvien.csv')

df1.shape

df1.info()

df2.shape

# Tạo danh sách các cột từ '_1' đến '_56'
columns_to_drop = [f'_{i}' for i in range(1, 57)]

# Bỏ các cột này
df2 = df2.drop(columns=columns_to_drop)

df2.info()

df_merged = pd.merge(df1, df2, on='mssv', how='outer')

df_merged.shape

df_merged.info()

df_merged.head(5)

Q1 = df_merged['sotchk'].quantile(0.25)
Q3 = df_merged['sotchk'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df_merged[(df_merged['sotchk'] >= lower_bound) & (df_merged['sotchk'] <= upper_bound)]

print(f"Số lượng hàng trước khi loại bỏ outlier: {len(df_merged)}")
print(f"Số lượng hàng sau khi loại bỏ outlier: {len(df_filtered)}")

# Sắp xếp mssv
df_filtered = df_filtered.sort_values(['mssv', 'namhoc', 'hocky'])

# Nhóm 'mssv' tạo'dtbhk2'
df_filtered['dtbhk2'] = df_filtered.groupby('mssv')['dtbhk'].shift(1)


df_filtered = df_filtered.dropna(subset=['dtbhk2']).reset_index(drop=True)

print(df_filtered[['mssv', 'dtbhk', 'dtbhk2']])

df_filtered.to_csv(r'C:\Steam Game\MXH\IS353.P12_Nhom4\data\processed\final_data.csv', index=False)
