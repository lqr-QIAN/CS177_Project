import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import time

def load_data():
    """加载response和scored数据"""
    print("Loading data...")
    # 设置数据路径
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    response_path = osp.join(data_root, 'raw_data/drugcombs_response.csv')
    scored_path = osp.join(data_root, 'raw_data/drugcombs_scored.csv')
    
    # 读取数据
    print("Reading response data...")
    response_df = pd.read_csv(response_path)
    print(f"Response data shape: {response_df.shape}")
    
    print("Reading scored data...")
    scored_df = pd.read_csv(scored_path)
    print(f"Scored data shape: {scored_df.shape}")
    
    return response_df, scored_df

def process_response_data(response_df, scored_df):
    """处理response数据，保留唯一的药物组合并添加cell_line信息"""
    print("\nProcessing response data...")
    
    # 创建总进度条
    total_steps = 4
    pbar = tqdm(total=total_steps, desc="Overall Progress", position=0)
    
    # 1. 首先对response数据按药物和浓度分组，每组只保留第一个block
    print("\nStep 1: Finding unique drug combinations...")
    step1_pbar = tqdm(total=100, desc="Finding unique combinations", position=1, leave=False)
    
    # 模拟进度
    for i in range(10):
        time.sleep(0.1)  # 模拟处理时间
        step1_pbar.update(10)
    
    unique_combinations = response_df.drop_duplicates(
        subset=['DrugRow', 'DrugCol', 'ConcRow', 'ConcCol'],
        keep='first'
    )
    step1_pbar.close()
    pbar.update(1)
    pbar.set_description("Step 1 completed")
    
    print(f"Unique combinations found: {len(unique_combinations)}")
    
    # 2. 重命名列
    print("\nStep 2: Renaming columns...")
    step2_pbar = tqdm(total=100, desc="Renaming columns", position=1, leave=False)
    
    # 模拟进度
    for i in range(10):
        time.sleep(0.05)  # 模拟处理时间
        step2_pbar.update(10)
    
    unique_combinations = unique_combinations.rename(columns={
        'DrugRow': 'Drug1',
        'DrugCol': 'Drug2',
        'ConcRow': 'Concentration1',
        'ConcCol': 'Concentration2'
    })
    step2_pbar.close()
    pbar.update(1)
    pbar.set_description("Step 2 completed")
    
    # 3. 添加cell_line信息
    print("\nStep 3: Adding cell line information...")
    step3_pbar = tqdm(total=100, desc="Adding cell line info", position=1, leave=False)
    
    # 模拟进度
    for i in range(10):
        time.sleep(0.05)  # 模拟处理时间
        step3_pbar.update(10)
    
    # 从scored数据中获取cell_line信息
    cell_line_map = scored_df[['BlockID', 'cell_line']].drop_duplicates()
    unique_combinations = unique_combinations.merge(
        cell_line_map,
        on='BlockID',
        how='left'
    )
    step3_pbar.close()
    pbar.update(1)
    pbar.set_description("Step 3 completed")
    
    # 4. 选择并重排列
    print("\nStep 4: Finalizing data...")
    step4_pbar = tqdm(total=100, desc="Finalizing data", position=1, leave=False)
    
    # 模拟进度
    for i in range(10):
        time.sleep(0.05)  # 模拟处理时间
        step4_pbar.update(10)
    
    final_columns = [
        'Drug1', 'Drug2', 'Concentration1', 'Concentration2', 
        'Response', 'BlockID', 'cell_line'
    ]
    unique_combinations = unique_combinations[final_columns]
    step4_pbar.close()
    pbar.update(1)
    pbar.set_description("Step 4 completed")
    pbar.close()
    
    return unique_combinations

def save_processed_data(processed_df, output_path):
    """保存处理后的数据"""
    print("\nSaving processed data...")
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建保存进度条
    save_pbar = tqdm(total=100, desc="Saving data")
    
    # 模拟保存进度
    for i in range(10):
        time.sleep(0.1)  # 模拟保存时间
        save_pbar.update(10)
    
    # 保存数据
    processed_df.to_csv(output_path, index=False)
    save_pbar.close()
    print(f"Processed data saved to: {output_path}")

def main():
    print("Starting response data processing...")
    # 1. 加载数据
    response_df, scored_df = load_data()
    
    # 2. 处理数据
    processed_df = process_response_data(response_df, scored_df)
    
    # 3. 保存处理后的数据
    output_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/drugcombs_response_unique.csv'
    save_processed_data(processed_df, output_path)
    
    # 4. 显示处理结果
    print("\nProcessing completed!")
    print(f"Original response data rows: {len(response_df)}")
    print(f"Processed data rows: {len(processed_df)}")
    print(f"Reduction ratio: {len(response_df)/len(processed_df):.2f}x")

if __name__ == '__main__':
    main() 
