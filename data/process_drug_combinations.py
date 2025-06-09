import pandas as pd

def process_drug_combinations():
    # 读取文件
    print("Reading files...")
    matched_df = pd.read_csv('data/processed/matched_drug_combinations.csv')
    mapped_df = pd.read_csv('data/processed/drugcombs_response_mapped.csv')
    
    # 选择需要的列并重新排序
    columns = ['Drug1_SMILES', 'Drug2_SMILES', 'depmap_id', 
              'Concentration1', 'Concentration2', 'Response']
    
    # 处理matched_drug_combinations.csv
    print("\nProcessing matched_drug_combinations.csv...")
    matched_processed = matched_df[columns].copy()
    # 重命名列
    matched_processed.columns = ['0', '1', '2', '3', '4', '5']
    matched_output = 'data/processed/matched_drug_combinations_processed.csv'
    matched_processed.to_csv(matched_output, index=False)
    print(f"Processed data saved to {matched_output}")
    print(f"Number of rows: {len(matched_processed)}")
    
    # 处理drugcombs_response_mapped.csv
    print("\nProcessing drugcombs_response_mapped.csv...")
    mapped_processed = mapped_df[columns].copy()
    # 重命名列
    mapped_processed.columns = ['0', '1', '2', '3', '4', '5']
    mapped_output = 'data/processed/drugcombs_response_mapped_processed.csv'
    mapped_processed.to_csv(mapped_output, index=False)
    print(f"Processed data saved to {mapped_output}")
    print(f"Number of rows: {len(mapped_processed)}")
    
    # 打印一些统计信息
    print("\nStatistics:")
    print("\nFor matched_drug_combinations_processed.csv:")
    print(f"Unique drug combinations: {len(matched_processed[['0', '1']].drop_duplicates())}")
    print(f"Unique cell lines: {matched_processed['2'].nunique()}")
    
    print("\nFor drugcombs_response_mapped_processed.csv:")
    print(f"Unique drug combinations: {len(mapped_processed[['0', '1']].drop_duplicates())}")
    print(f"Unique cell lines: {mapped_processed['2'].nunique()}")

if __name__ == '__main__':
    process_drug_combinations() 
