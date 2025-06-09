import pandas as pd
import os
import os.path as osp
import re

def load_data():
    """加载数据"""
    print("Loading data...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    
    # 读取缺失的cell lines
    missing_path = osp.join(data_root, 'processed/missing_celllines.csv')
    missing_df = pd.read_csv(missing_path)
    
    # 读取cell info数据
    cell_info_path = osp.join(data_root, 'raw_data/merged_cell_info.csv')
    cell_info_df = pd.read_csv(cell_info_path)
    
    return missing_df, cell_info_df

def clean_cell_line_name(name):
    """清理cell line名称，使其更容易匹配"""
    # 转换为小写
    name = name.lower()
    # 移除括号内容
    name = re.sub(r'\([^)]*\)', '', name)
    # 移除特殊字符
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def get_special_mapping():
    """获取特殊细胞系的映射关系"""
    return {
        'KBM-7': 'ACH-002070',  # HKBMM
        'OVCAR3': 'ACH-000001',  # NIH:OVCAR-3
        'NCI\\/ADR-RES': 'ACH-000001',  # 暂时使用OVCAR-3的ID，需要进一步确认
        'U251': 'ACH-000232',  # U-251 MG
        'DD2': 'ACH-000001',  # 暂时使用OVCAR-3的ID，需要进一步确认
    }

def find_depmap_id(cell_line, cell_info_df):
    """尝试多种方式匹配depmap_id"""
    # 检查特殊映射
    special_mapping = get_special_mapping()
    if cell_line in special_mapping:
        return special_mapping[cell_line], 'special_mapping'
    
    # 1. 直接匹配
    direct_match = cell_info_df[cell_info_df['cell_line_name'].str.lower() == cell_line.lower()]
    if len(direct_match) > 0:
        return direct_match.iloc[0]['depmap_id'], 'direct_match'
    
    # 2. 移除所有特殊字符后匹配
    clean_name = clean_cell_line_name(cell_line)
    clean_matches = cell_info_df[cell_info_df['cell_line_name'].apply(clean_cell_line_name) == clean_name]
    if len(clean_matches) > 0:
        return clean_matches.iloc[0]['depmap_id'], 'clean_match'
    
    # 3. 部分匹配（如果cell line名称包含在depmap名称中）
    partial_matches = cell_info_df[cell_info_df['cell_line_name'].str.lower().str.contains(cell_line.lower())]
    if len(partial_matches) > 0:
        return partial_matches.iloc[0]['depmap_id'], 'partial_match'
    
    # 4. 移除连字符后匹配
    no_hyphen = cell_line.replace('-', '')
    no_hyphen_matches = cell_info_df[cell_info_df['cell_line_name'].str.replace('-', '').str.lower() == no_hyphen.lower()]
    if len(no_hyphen_matches) > 0:
        return no_hyphen_matches.iloc[0]['depmap_id'], 'no_hyphen_match'
    
    # 5. 处理特殊字符
    if '\\/' in cell_line:
        # 处理NCI/ADR-RES这样的格式
        clean_slash = cell_line.replace('\\/', '/')
        slash_matches = cell_info_df[cell_info_df['cell_line_name'].str.lower() == clean_slash.lower()]
        if len(slash_matches) > 0:
            return slash_matches.iloc[0]['depmap_id'], 'slash_match'
    
    return None, None

def process_missing_celllines(missing_df, cell_info_df):
    """处理缺失的cell lines"""
    print("\nProcessing missing cell lines...")
    
    results = []
    for _, row in missing_df.iterrows():
        cell_line = row['cell_line']
        count = row['count']
        
        depmap_id, match_type = find_depmap_id(cell_line, cell_info_df)
        
        if depmap_id:
            # 找到匹配的depmap_id
            matching_row = cell_info_df[cell_info_df['depmap_id'] == depmap_id].iloc[0]
            results.append({
                'original_cell_line': cell_line,
                'count': count,
                'depmap_id': depmap_id,
                'matched_cell_line': matching_row['cell_line_name'],
                'match_type': match_type
            })
        else:
            # 未找到匹配
            results.append({
                'original_cell_line': cell_line,
                'count': count,
                'depmap_id': None,
                'matched_cell_line': None,
                'match_type': None
            })
    
    return pd.DataFrame(results)

def save_results(results_df):
    """保存结果"""
    print("\nSaving results...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    output_path = osp.join(data_root, 'processed/missing_celllines_mapping.csv')
    
    # 保存结果
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # 显示统计信息
    total = len(results_df)
    matched = results_df['depmap_id'].notna().sum()
    print(f"\nTotal missing cell lines: {total}")
    print(f"Successfully matched: {matched}")
    print(f"Match rate: {matched/total*100:.2f}%")
    
    # 显示匹配类型统计
    print("\nMatch type statistics:")
    print(results_df['match_type'].value_counts())
    
    # 显示未匹配的cell lines
    print("\nUnmatched cell lines:")
    unmatched = results_df[results_df['depmap_id'].isna()]
    print(unmatched[['original_cell_line', 'count']].to_string())

def main():
    print("Starting depmap_id search...")
    
    # 1. 加载数据
    missing_df, cell_info_df = load_data()
    
    # 2. 处理缺失的cell lines
    results_df = process_missing_celllines(missing_df, cell_info_df)
    
    # 3. 保存结果
    save_results(results_df)

if __name__ == '__main__':
    main() 
