import numpy as np
import os

# ========== 参数 ==========
all_items_path = '/root/lanyun-tmp/Project/SynergyX/data/split/all_items_new.npy'  # 你的 all_items1.npy 路径
split_dir = '/root/lanyun-tmp/Project/SynergyX/data/split'  # 输出目录
n_fold = 0  # 折数编号，可根据需要调整

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ========== 加载数据 ==========
all_items = np.load(all_items_path, allow_pickle=True)
print(f"✅ 加载 all_items1.npy，样本数: {len(all_items)}")

# ========== 打乱并划分 ==========
np.random.seed(42)
indices = np.random.permutation(len(all_items))

n_train = int(train_ratio * len(all_items))
n_val = int(val_ratio * len(all_items))
n_test = len(all_items) - n_train - n_val

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

train_items = all_items[train_indices]
val_items = all_items[val_indices]
test_items = all_items[test_indices]

print(f"✅ 划分结果: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

# ========== 保存 ==========
os.makedirs(split_dir, exist_ok=True)
np.save(os.path.join(split_dir, f'{n_fold}_fold_tr_items_new.npy'), train_items)
np.save(os.path.join(split_dir, f'{n_fold}_fold_val_items_new.npy'), val_items)
np.save(os.path.join(split_dir, f'{n_fold}_fold_test_items_new.npy'), test_items)

print(f"✅ 保存完成！文件路径:")
print(f"- {split_dir}/{n_fold}_fold_tr_items2.npy")
print(f"- {split_dir}/{n_fold}_fold_val_items2.npy")
print(f"- {split_dir}/{n_fold}_fold_test_items2.npy")
