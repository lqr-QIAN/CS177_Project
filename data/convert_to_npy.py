import pandas as pd
import numpy as np

def compare_arrays(arr1, arr2, name):
    """比较两个数组并打印详细信息"""
    if arr1.shape != arr2.shape:
        print(f"\n{name} shape mismatch:")
        print(f"Original shape: {arr1.shape}")
        print(f"Loaded shape: {arr2.shape}")
        return False
    
    # 检查每个元素
    for i in range(min(5, len(arr1))):  # 只检查前5行
        for j in range(arr1.shape[1]):
            if arr1[i,j] != arr2[i,j]:
                print(f"\n{name} mismatch at position [{i},{j}]:")
                print(f"Original: {arr1[i,j]} (type: {type(arr1[i,j])})")
                print(f"Loaded: {arr2[i,j]} (type: {type(arr2[i,j])})")
                return False
    
    return True

def convert_to_npy():
    # 读取CSV文件
    print("Reading CSV files...")
    mapped_df = pd.read_csv('data/processed/drugcombs_response_mapped_processed.csv')
    matched_df = pd.read_csv('data/processed/matched_drug_combinations_processed.csv')
    
    # 转换为numpy数组
    print("\nConverting to numpy arrays...")
    mapped_array = mapped_df.values
    matched_array = matched_df.values
    
    # 打印数据类型信息
    print("\nData type information:")
    print("\nFor mapped_array:")
    for i, col in enumerate(mapped_df.columns):
        print(f"Column {i} ({col}): {mapped_array[:,i].dtype}")
    
    print("\nFor matched_array:")
    for i, col in enumerate(matched_df.columns):
        print(f"Column {i} ({col}): {matched_array[:,i].dtype}")
    
    # 保存为npy文件
    print("\nSaving numpy arrays...")
    np.save('data/processed/drugcombs_response_mapped_processed.npy', mapped_array, allow_pickle=True)
    np.save('data/processed/matched_drug_combinations_processed.npy', matched_array, allow_pickle=True)
    
    # 打印信息
    print("\nFile information:")
    print("\nFor drugcombs_response_mapped_processed.npy:")
    print(f"Shape: {mapped_array.shape}")
    print(f"Data type: {mapped_array.dtype}")
    
    print("\nFor matched_drug_combinations_processed.npy:")
    print(f"Shape: {matched_array.shape}")
    print(f"Data type: {matched_array.dtype}")
    
    # 验证数据
    print("\nVerifying data...")
    # 加载npy文件并比较
    mapped_loaded = np.load('data/processed/drugcombs_response_mapped_processed.npy', allow_pickle=True)
    matched_loaded = np.load('data/processed/matched_drug_combinations_processed.npy', allow_pickle=True)
    
    print("\nVerification results:")
    mapped_verified = compare_arrays(mapped_array, mapped_loaded, "drugcombs_response_mapped_processed.npy")
    matched_verified = compare_arrays(matched_array, matched_loaded, "matched_drug_combinations_processed.npy")
    
    print(f"\nFinal verification:")
    print(f"drugcombs_response_mapped_processed.npy verification: {mapped_verified}")
    print(f"matched_drug_combinations_processed.npy verification: {matched_verified}")

if __name__ == '__main__':
    convert_to_npy() 
