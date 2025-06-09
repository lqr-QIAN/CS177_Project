import pandas as pd
import os
import os.path as osp
from tqdm import tqdm

def load_data():
    """加载response和scored数据"""
    print("Loading data...")
    # 设置数据路径
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    response_path = osp.join(data_root, 'processed/drugcombs_response_unique.csv')
    scored_path = osp.join(data_root, 'raw_data/drugcombs_scored.csv')
    
    # 读取数据
    print("Reading response data...")
    response_df = pd.read_csv(response_path)
    print(f"Response data shape: {response_df.shape}")
    
    print("Reading scored data...")
    scored_df = pd.read_csv(scored_path)
    print(f"Scored data shape: {scored_df.shape}")
    
    return response_df, scored_df

def process_data(response_df, scored_df):
    """处理数据，添加cell_line信息并删除不需要的列"""
    print("\nProcessing data...")
    
    # 从scored数据中获取cell_line信息
    print("Adding cell line information...")
    cell_line_map = scored_df[['ID', 'Cell line']].drop_duplicates()
    cell_line_map = cell_line_map.rename(columns={'ID': 'BlockID', 'Cell line': 'cell_line'})
    processed_df = response_df.merge(
        cell_line_map,
        on='BlockID',
        how='left'
    )
    
    # 选择并重排列
    print("Finalizing data...")
    final_columns = [
        'Drug1', 'Drug2', 'Concentration1', 'Concentration2', 
        'Response', 'BlockID', 'cell_line'
    ]
    processed_df = processed_df[final_columns]
    
    return processed_df

def save_processed_data(processed_df, output_path):
    """保存处理后的数据"""
    print("\nSaving processed data...")
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

def main():
    print("Starting data processing...")
    # 1. 加载数据
    response_df, scored_df = load_data()
    
    # 2. 处理数据
    processed_df = process_data(response_df, scored_df)
    
    # 3. 保存处理后的数据
    output_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/drugcombs_response_with_cellline.csv'
    save_processed_data(processed_df, output_path)
    
    # 4. 显示处理结果
    print("\nProcessing completed!")
    print(f"Original response data rows: {len(response_df)}")
    print(f"Processed data rows: {len(processed_df)}")
    print(f"Number of unique cell lines: {processed_df['cell_line'].nunique()}")

if __name__ == '__main__':
    main() 
