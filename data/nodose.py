import numpy as np
import os

def convert_to_synergyx_format(input_path, output_path):
    items = np.load(input_path, allow_pickle=True)
    new_items = []
    for item in items:
        if len(item) == 6:
            drugA, drugB, cell_line, doseA, doseB, response = item
            new_items.append([drugA, drugB, cell_line, response])
        else:
            print("⚠️ Unexpected item format:", item)

    new_items = np.array(new_items, dtype=object)
    np.save(output_path, new_items)
    print(f"Saved converted file to {output_path} with {len(new_items)} samples.")

# 示例调用
data_root = '/root/lanyun-tmp/Project/SynergyX/data'
n_fold = 0

convert_to_synergyx_format(
    input_path=os.path.join(data_root, f'split/{n_fold}_fold_tr_items.npy'),
    output_path=os.path.join(data_root, f'split/{n_fold}_fold_tr_items_nodose.npy')
)

convert_to_synergyx_format(
    input_path=os.path.join(data_root, f'split/{n_fold}_fold_val_items.npy'),
    output_path=os.path.join(data_root, f'split/{n_fold}_fold_val_items_nodose.npy')
)

convert_to_synergyx_format(
    input_path=os.path.join(data_root, f'split/{n_fold}_fold_test_items.npy'),
    output_path=os.path.join(data_root, f'split/{n_fold}_fold_test_items_nodose.npy')
)
