import pandas as pd
import os

# 设置文件路径
drug_info_extracted_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/drug_info_extracted.csv'
drug_smiles_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/drug_smiles.csv'
drug_names_canonical_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/drug_names_canonical_smiles.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_drug_info.csv'
duplicates_output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/duplicates_drug_info.csv'
selected_duplicates_output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/selected_duplicates_drug_info.csv'

def process_duplicates(df):
    """处理重复数据，返回处理后的数据框和统计信息"""
    print("\n开始处理重复数据...")
    
    # 1. 删除存在空缺的行
    original_len = len(df)
    df = df.dropna(subset=['dname', 'canonical_smiles'])
    print(f"\n删除空缺值后的统计:")
    print(f"- 原始行数: {original_len}")
    print(f"- 删除空缺后行数: {len(df)}")
    print(f"- 删除的行数: {original_len - len(df)}")
    
    # 2. 计算每个SMILES被多少不同的药物使用
    smiles_usage = df.groupby('canonical_smiles')['dname'].nunique()
    df['smiles_usage_count'] = df['canonical_smiles'].map(smiles_usage)
    
    # 3. 计算SMILES长度
    df['smiles_length'] = df['canonical_smiles'].str.len()
    
    # 4. 对每个药物名称，按优先级选择SMILES
    # 优先级：1. smiles_usage_count最小（不共享的优先） 2. smiles_length最小（较短的优先）
    df = df.sort_values(['dname', 'smiles_usage_count', 'smiles_length'])
    final_df = df.drop_duplicates(subset=['dname'], keep='first')
    
    # 5. 收集统计信息
    stats = {
        'original_count': original_len,
        'after_dropna_count': len(df),
        'final_count': len(final_df),
        'unique_drugs': df['dname'].nunique(),
        'unique_smiles': df['canonical_smiles'].nunique(),
        'shared_smiles_count': len(smiles_usage[smiles_usage > 1]),
        'duplicate_drugs': df['dname'].value_counts()[df['dname'].value_counts() > 1].index.tolist()
    }
    
    # 6. 保存重复数据供参考
    duplicates_df = df[df['dname'].isin(stats['duplicate_drugs'])].copy()
    duplicates_df = duplicates_df.sort_values(['dname', 'smiles_usage_count', 'smiles_length'])
    duplicates_df.to_csv(duplicates_output_path, index=False)
    
    # 7. 保存被选中的重复药物数据
    selected_duplicates_df = final_df[final_df['dname'].isin(stats['duplicate_drugs'])].copy()
    selected_duplicates_df = selected_duplicates_df.sort_values('dname')
    selected_duplicates_df.to_csv(selected_duplicates_output_path, index=False)
    
    # 8. 删除临时列
    final_df = final_df.drop(['smiles_usage_count', 'smiles_length'], axis=1)
    
    return final_df, stats, duplicates_df, selected_duplicates_df

def merge_drug_info():
    print("开始合并药物信息文件...")
    
    # 读取三个文件
    try:
        print("读取 drug_info_extracted.csv...")
        df1 = pd.read_csv(drug_info_extracted_path)
        print(f"✅ 成功读取，共 {len(df1)} 条记录")
        
        print("\n读取 drug_smiles.csv...")
        df2 = pd.read_csv(drug_smiles_path)
        print(f"✅ 成功读取，共 {len(df2)} 条记录")
        
        print("\n读取 drug_names_canonical_smiles.csv...")
        df3 = pd.read_csv(drug_names_canonical_path)
        print(f"✅ 成功读取，共 {len(df3)} 条记录")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 合并数据框
    try:
        # 使用 dname 和 canonical_smiles 作为键合并
        merged_df = pd.concat([df1, df2, df3], ignore_index=True)
        # 删除重复行
        merged_df = merged_df.drop_duplicates(subset=['dname', 'canonical_smiles'])
        print(f"\n✅ 合并完成，去重后共 {len(merged_df)} 条记录")
    except Exception as e:
        print(f"❌ 合并数据失败: {e}")
        return
    
    # 处理重复数据
    try:
        final_df, stats, duplicates_df, selected_duplicates_df = process_duplicates(merged_df)
        
        # 保存最终结果
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ 成功保存处理后的数据到: {output_path}")
        
        # 打印详细统计信息
        print("\n数据统计:")
        print(f"- 原始记录数: {stats['original_count']}")
        print(f"- 删除空缺值后记录数: {stats['after_dropna_count']}")
        print(f"- 最终记录数: {stats['final_count']}")
        print(f"- 唯一药物名称数: {stats['unique_drugs']}")
        print(f"- 唯一SMILES数: {stats['unique_smiles']}")
        print(f"- 被多个药物共享的SMILES数: {stats['shared_smiles_count']}")
        print(f"- 重复的药物名称数: {len(stats['duplicate_drugs'])}")
        print(f"- 被选中的重复药物数: {len(selected_duplicates_df)}")
        
        # 打印一些示例
        if stats['duplicate_drugs']:
            print("\n重复药物处理示例:")
            example_drug = stats['duplicate_drugs'][0]
            print(f"\n药物 '{example_drug}' 的处理情况:")
            print("处理前:")
            print(duplicates_df[duplicates_df['dname'] == example_drug][['dname', 'canonical_smiles', 'smiles_usage_count', 'smiles_length']].to_string())
            print("\n处理后:")
            print(selected_duplicates_df[selected_duplicates_df['dname'] == example_drug][['dname', 'canonical_smiles']].to_string())
        
    except Exception as e:
        print(f"❌ 处理数据失败: {e}")
        return

if __name__ == "__main__":
    merge_drug_info() 
