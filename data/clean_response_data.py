import pandas as pd

def clean_response_data():
    # 读取CSV文件
    print("Reading CSV file...")
    df = pd.read_csv('data/processed/drugcombs_response_mapped_processed.csv')
    
    print("\nOriginal data shape:", df.shape)
    print("\nColumns in the file:")
    for col in df.columns:
        print(f"- {col}")
    
    # 删除任何包含空值的行
    print("\nRemoving rows with missing values...")
    df_cleaned = df.dropna()
    
    # 保存清理后的数据
    print("\nSaving cleaned data...")
    df_cleaned.to_csv('data/processed/drugcombs_response_mapped_processed_cleaned.csv', index=False)
    
    # 打印统计信息
    print("\nResults:")
    print(f"Original number of rows: {df.shape[0]}")
    print(f"Number of rows after cleaning: {df_cleaned.shape[0]}")
    print(f"Number of rows removed: {df.shape[0] - df_cleaned.shape[0]}")
    
    # 检查每列的空值数量
    print("\nMissing values per column in original data:")
    missing_values = df.isnull().sum()
    for col, count in missing_values.items():
        if count > 0:
            print(f"{col}: {count} missing values")

if __name__ == '__main__':
    clean_response_data() 
