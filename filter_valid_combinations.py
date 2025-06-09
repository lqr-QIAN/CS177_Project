import pandas as pd

def filter_valid_combinations():
    print("开始筛选有效的药物组合数据...")
    
    try:
        # 读取数据文件
        print("读取数据文件...")
        response_df = pd.read_csv('/root/lanyun-tmp/Project/SynergyX/data/processed/drugcombs_response_with_cellline.csv')
        drug_info_df = pd.read_csv('/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_drug_info.csv')
        cell_info_df = pd.read_csv('/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_cell_info.csv')
        
        print(f"✅ 成功读取数据文件")
        print(f"- 药物组合数据: {len(response_df)} 行")
        print(f"- 药物信息数据: {len(drug_info_df)} 行")
        print(f"- 细胞系信息数据: {len(cell_info_df)} 行")
        
        # 获取有效的药物名称和细胞系名称列表
        valid_drugs = set(drug_info_df['dname'].str.strip().str.upper())
        valid_cell_lines = set(cell_info_df['cell_line_name'].str.strip().str.upper())
        
        # 标准化药物组合数据中的名称
        response_df['Drug1'] = response_df['Drug1'].str.strip().str.upper()
        response_df['Drug2'] = response_df['Drug2'].str.strip().str.upper()
        response_df['cell_line'] = response_df['cell_line'].str.strip().str.upper()
        
        # 筛选条件
        valid_combinations = response_df[
            (response_df['Drug1'].isin(valid_drugs)) & 
            (response_df['Drug2'].isin(valid_drugs)) & 
            (response_df['cell_line'].isin(valid_cell_lines))
        ]
        
        # 保存结果
        output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/valid_drug_combinations.csv'
        valid_combinations.to_csv(output_path, index=False)
        
        # 打印统计信息
        print("\n筛选结果统计:")
        print(f"- 原始数据行数: {len(response_df)}")
        print(f"- 有效组合行数: {len(valid_combinations)}")
        print(f"- 过滤掉的行数: {len(response_df) - len(valid_combinations)}")
        
        # 分析被过滤掉的原因
        invalid_drug1 = response_df[~response_df['Drug1'].isin(valid_drugs)]['Drug1'].unique()
        invalid_drug2 = response_df[~response_df['Drug2'].isin(valid_drugs)]['Drug2'].unique()
        invalid_cell_lines = response_df[~response_df['cell_line'].isin(valid_cell_lines)]['cell_line'].unique()
        
        print("\n无效数据统计:")
        print(f"- 无效的Drug1数量: {len(invalid_drug1)}")
        print(f"- 无效的Drug2数量: {len(invalid_drug2)}")
        print(f"- 无效的细胞系数量: {len(invalid_cell_lines)}")
        
        print(f"\n✅ 已将有效组合保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return

if __name__ == "__main__":
    filter_valid_combinations() 
