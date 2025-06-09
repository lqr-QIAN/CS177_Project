import pandas as pd
import os
import os.path as osp

def load_data():
    """加载数据"""
    print("Loading data...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    
    # 读取原始数据
    response_path = osp.join(data_root, 'processed/drugcombs_response_with_cellline.csv')
    response_df = pd.read_csv(response_path)
    
    # 读取映射后的数据
    mapped_path = osp.join(data_root, 'processed/drugcombs_response_mapped.csv')
    mapped_df = pd.read_csv(mapped_path)
    
    return response_df, mapped_df

def find_missing_celllines(response_df, mapped_df):
    """找出未找到depmap_id的cell lines"""
    print("\nFinding missing cell lines...")
    
    # 获取所有唯一的cell lines
    all_celllines = response_df['cell_line'].unique()
    
    # 获取未找到depmap_id的cell lines
    missing_celllines = []
    for cell_line in all_celllines:
        # 检查这个cell line是否有任何一行成功映射到depmap_id
        has_mapping = mapped_df[mapped_df['cell_line'] == cell_line]['depmap_id'].notna().any()
        if not has_mapping:
            # 计算这个cell line在数据中出现的次数
            count = len(response_df[response_df['cell_line'] == cell_line])
            missing_celllines.append({
                'cell_line': cell_line,
                'count': count
            })
    
    # 转换为DataFrame并排序
    missing_df = pd.DataFrame(missing_celllines)
    missing_df = missing_df.sort_values('count', ascending=False)
    
    return missing_df

def save_results(missing_df):
    """保存结果"""
    print("\nSaving results...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    output_path = osp.join(data_root, 'processed/missing_celllines.csv')
    
    # 保存结果
    missing_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # 显示统计信息
    print(f"\nTotal number of unique cell lines: {len(missing_df)}")
    print(f"Total number of data points affected: {missing_df['count'].sum()}")
    print("\nTop 10 most frequent missing cell lines:")
    print(missing_df.head(10).to_string())

def main():
    print("Starting analysis...")
    
    # 1. 加载数据
    response_df, mapped_df = load_data()
    
    # 2. 找出缺失的cell lines
    missing_df = find_missing_celllines(response_df, mapped_df)
    
    # 3. 保存结果
    save_results(missing_df)

if __name__ == '__main__':
    main() 
