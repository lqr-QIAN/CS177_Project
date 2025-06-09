import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset
import psutil


class MyInMemoryDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 dgi_data=None,
                 transform=None,
                 pre_transform=None,
                 args=None,
                 max_node_num=155,
                 force_reprocess=False):  # ✅ 新增参数

        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)

        if args.celldataset == 1:
            self.name = osp.basename(data_items).split('items')[0] + '18498g'
        elif args.celldataset == 2:
            self.name = osp.basename(data_items).split('items')[0] + '4079g'
        elif args.celldataset == 3:
            self.name = osp.basename(data_items).split('items')[0] + '963g'

        self.name = self.name + '_TransDrug_norm'

        if args.mode == 'infer':
            self.name = osp.basename(data_items).split('items')[0]

        self.args = args
        self.data_items = np.load(data_items, allow_pickle=True)
        print(f"[Debug] Loaded {len(self.data_items)} items from {data_items}")

        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.drugs = np.load(drugs_data, allow_pickle=True).item()
        if dgi_data:
            self.dgi = np.load(dgi_data, allow_pickle=True).item()
        else:
            self.dgi = {}

        self.max_node_num = max_node_num

        if force_reprocess or not os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data not found or force_reprocess=True. Doing pre-processing...')
            self.process()
        else:
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []
        data_len = len(self.data_items)
        print(f"Starting to process {data_len} items...")
        
        # 监控内存使用
        process = psutil.Process()

        for i in tqdm(range(data_len)):
            if i % 10000 == 0:
                print(f"Processing item {i}, Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            try:
                drugA, drugB, c1, label = self.data_items[i]
                cell_features = self.celllines[c1]

                dgiA = self.dgi.get(drugA, np.ones(cell_features.shape[0]))
                dgiB = self.dgi.get(drugB, np.ones(cell_features.shape[0]))

                drugA_features = self.drugs[drugA]
                drugB_features = self.drugs[drugB]

                cell_drug_data = Data()
                cell_drug_data.drugA = torch.Tensor(np.array([drugA_features])).to(dtype=torch.float16)
                cell_drug_data.drugB = torch.Tensor(np.array([drugB_features])).to(dtype=torch.float16)
                cell_drug_data.x_cell = torch.as_tensor(cell_features).to(dtype=torch.float16)
                cell_drug_data.y = torch.Tensor([float(label)]).to(dtype=torch.float16)
                cell_drug_data.dgiA = torch.Tensor(dgiA).to(dtype=torch.float16)
                cell_drug_data.dgiB = torch.Tensor(dgiB).to(dtype=torch.float16)

                data_list.append(cell_drug_data)
            
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue

        print(f"Processed {len(data_list)} items successfully")
        print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f'Graph construction done. {len(data_list)} samples processed. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Dataset construction done.')
