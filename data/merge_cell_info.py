import pandas as pd
import os

# 设置文件路径
cell_info_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/cell_info_extracted.csv'
merged_cell_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_cell_lines_cleaned.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_cell_info.csv'
duplicates_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/duplicate_depmap_ids.csv'
selected_records_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/selected_duplicate_cell_info.csv'

def merge_cell_info():
    print("开始合并细胞系信息...")
    
    try:
        # 读取两个文件
        print("读取 cell_info_extracted.csv...")
        cell_info_df = pd.read_csv(cell_info_path)
        print(f"✅ 成功读取，共 {len(cell_info_df)} 条记录")
        
        print("\n读取 merged_cell_lines_cleaned.csv...")
        merged_cell_df = pd.read_csv(merged_cell_path)
        print(f"✅ 成功读取，共 {len(merged_cell_df)} 条记录")
        
        # 显示两个文件的列名
        print("\n文件列名信息:")
        print("\ncell_info_extracted.csv 的列:")
        print(cell_info_df.columns.tolist())
        print("\nmerged_cell_lines_cleaned.csv 的列:")
        print(merged_cell_df.columns.tolist())
        
        # 合并数据
        print("\n合并数据...")
        # 使用 outer join 保留所有记录
        merged_df = pd.merge(
            cell_info_df,
            merged_cell_df,
            on=['depmap_id', 'cell_line_name'],
            how='outer',
            indicator=True
        )
        
        # 显示合并结果统计
        print("\n合并结果统计:")
        print(f"- 合并后总记录数: {len(merged_df)}")
        print("\n合并类型统计:")
        merge_stats = merged_df['_merge'].value_counts()
        print(merge_stats)
        print("\n合并类型说明:")
        print("- left_only: 仅在 cell_info_extracted.csv 中存在的记录")
        print("- right_only: 仅在 merged_cell_lines_cleaned.csv 中存在的记录")
        print("- both: 在两个文件中都存在的记录")
        
        # 删除辅助列
        merged_df = merged_df.drop('_merge', axis=1)
        
        # 删除depmap_id或cell_line_name为空的行
        print("\n删除depmap_id或cell_line_name为空的行...")
        original_len = len(merged_df)
        merged_df = merged_df.dropna(subset=['depmap_id', 'cell_line_name'])
        print(f"- 原始行数: {original_len}")
        print(f"- 删除空值后行数: {len(merged_df)}")
        print(f"- 删除的行数: {original_len - len(merged_df)}")
        
        # 处理重复的depmap_id
        print("\n处理重复的depmap_id...")
        # 添加cell_line_name长度列
        merged_df['name_length'] = merged_df['cell_line_name'].str.len()
        
        # 找出所有重复的depmap_id
        depmap_counts = merged_df['depmap_id'].value_counts()
        duplicate_depmaps = depmap_counts[depmap_counts > 1]
        
        if len(duplicate_depmaps) > 0:
            print(f"\n发现 {len(duplicate_depmaps)} 个重复的depmap_id")
            
            # 获取所有重复depmap_id的原始记录
            duplicate_records = merged_df[merged_df['depmap_id'].isin(duplicate_depmaps.index)]
            duplicate_records = duplicate_records.sort_values(['depmap_id', 'name_length'], ascending=[True, False])
            duplicate_records.to_csv(duplicates_path, index=False)
            print(f"已将重复记录保存到: {duplicates_path}")
            
            # 获取被选择保留的记录（每个depmap_id中name_length最长的记录）
            selected_records = merged_df.sort_values('name_length', ascending=False).groupby('depmap_id').first().reset_index()
            selected_records = selected_records[selected_records['depmap_id'].isin(duplicate_depmaps.index)]
            selected_records = selected_records.sort_values('depmap_id')
            selected_records.to_csv(selected_records_path, index=False)
            print(f"已将选择保留的记录保存到: {selected_records_path}")
            
            print("\n重复depmap_id的统计:")
            print(duplicate_depmaps)
        else:
            print("\n没有发现重复的depmap_id")
        
        # 按depmap_id分组，选择cell_line_name最长的记录
        merged_df = merged_df.sort_values('name_length', ascending=False).groupby('depmap_id').first().reset_index()
        
        # 删除辅助列
        merged_df = merged_df.drop('name_length', axis=1)
        
        # 保存结果
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ 成功保存合并数据到: {output_path}")
        
        # 打印最终统计信息
        print("\n最终数据统计:")
        print(f"- 总记录数: {len(merged_df)}")
        print(f"- 唯一 depmap_id 数: {merged_df['depmap_id'].nunique()}")
        print(f"- 唯一 cell_line_name 数: {merged_df['cell_line_name'].nunique()}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return

if __name__ == "__main__":
    merge_cell_info() 
