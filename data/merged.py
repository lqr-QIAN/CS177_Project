import numpy as np
import pandas as pd

# 文件路径
all_items_paths = [
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items1.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items11.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items2.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items3.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items4.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items5.npy',
    '/root/lanyun-tmp/Project/SynergyX/data/split/all_items7.npy'
]

output_path = '/root/lanyun-tmp/Project/SynergyX/data/split/all_items_merged1.npy'

# 加载并合并所有数据
all_items_list = []
for path in all_items_paths:
    data = np.load(path, allow_pickle=True)
    print(f"✅ 加载 {path}，共 {len(data)} 条记录")
    all_items_list.append(data)

all_items = np.concatenate(all_items_list, axis=0)

# 转为 DataFrame 便于去重
all_items_df = pd.DataFrame(all_items.tolist())

# 去重
all_items_df = all_items_df.drop_duplicates()

print(f"✅ 合并并去重后样本数：{len(all_items_df)}")

# 保存
all_items_merged = all_items_df.to_numpy(dtype=object)
np.save(output_path, all_items_merged)

print(f"✅ 合并并去重后的文件已保存为 {output_path}")
