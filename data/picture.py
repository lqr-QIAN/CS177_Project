import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 主目录路径
base_dir = '/root/lanyun-tmp/Project/SynergyX/cell_line_items'

# 获取所有细胞系目录
cell_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"📊 找到 {len(cell_dirs)} 个细胞系目录")

# 遍历所有子目录
for cell_name in tqdm(cell_dirs, desc="处理细胞系", ncols=100):
    cell_folder = os.path.join(base_dir, cell_name)
    csv_path = os.path.join(cell_folder, f'{cell_name}_items.csv')
    
    if not os.path.exists(csv_path):
        tqdm.write(f"❌ 未找到 {cell_name}_items.csv，跳过")
        continue

    try:
        df = pd.read_csv(csv_path, header=None)  # 无表头，按列索引读入
        if df.shape[1] < 6:
            tqdm.write(f"⚠️  {cell_name} 数据列不足，跳过")
            continue

        df.columns = ['drug1', 'drug2', 'cell', 'dose1', 'dose2', 'label']
        
        # 获取药物组合数量
        drug_pairs = list(df.groupby(['drug1', 'drug2']))
        tqdm.write(f"📈 {cell_name}: 处理 {len(drug_pairs)} 个药物组合")

        # 按药物组合分组画图
        for i, ((drug1, drug2), group) in enumerate(drug_pairs):
            plt.figure()
            plt.scatter(group['dose1'], group['label'], label=f'Drug 1', alpha=0.6)
            plt.scatter(group['dose2'], group['label'], label=f'Drug 2', alpha=0.6)
            plt.xlabel('Dose')
            plt.ylabel('Response')
            plt.title(f'{cell_name} - Pair {i+1}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(cell_folder, f'pair_{i+1}.png')
            plt.savefig(save_path)
            plt.close()

    except Exception as e:
        tqdm.write(f"❌ {cell_name} 处理失败：{e}")

print("\n✅ 所有细胞系的剂量-反应图已生成并保存在各自文件夹中。")
