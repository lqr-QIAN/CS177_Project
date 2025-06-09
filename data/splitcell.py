import os
import numpy as np
import pandas as pd

# 输入文件路径
input_file = '/root/lanyun-tmp/Project/SynergyX/data/split/all_items_merged1.npy'  # 改为实际路径
output_dir = 'cell_line_items'

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data = np.load(input_file, allow_pickle=True)
df = pd.DataFrame(data.tolist())

# 确认列名（假设第三列为 cell_line）
cell_column_name = df.columns[2]  # 默认第3列是cell，如需修改请手动设定：'cell_ID'或'sample_id'等

# 分组保存
for cell_name, group in df.groupby(cell_column_name):
    safe_cell_name = cell_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'all_{safe_cell_name}_items.npy')
    np.save(output_path, group.to_numpy())

print(f'保存完成，共保存 {df[cell_column_name].nunique()} 个细胞系文件至 {output_dir}/')
