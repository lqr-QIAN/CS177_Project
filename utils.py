import os
import os.path as osp
import random
import numpy as np
import torch
from mmcv.utils import collect_env as collect_base_env
from torch_geometric.loader import DataLoader
from dataset.My_inMemory_dataset_dose2 import MyInMemoryDataset
from metrics import get_metrics
from tqdm import tqdm

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=3, filename=None, metric=None, n_fold=None, folder=None, min_delta=0.0):
        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                f"Invalid metric: {metric}"
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print(f'For metric {metric}, the higher the better')
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print(f'For metric {metric}, the lower the better')
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

        self._check = self._check_higher if mode == 'higher' else self._check_lower

    def _check_higher(self, score, prev_best_score):
        return (score - prev_best_score) > self.min_delta

    def _check_lower(self, score, prev_best_score):
        return (prev_best_score - score) > self.min_delta

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))


    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info


def load_dataloader(n_fold,args):
    work_dir = args.workdir
    data_root = osp.join(work_dir,'data')

    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,'/root/lanyun-tmp/Project/SynergyX/data/0_cell_data/4079g/4cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes.npy')

    
    drugs_data = osp.join(data_root,'/root/lanyun-tmp/Project/SynergyX/data/1_drug_data/4drugSmile_drugSubEmbed_canonical.npy')
    """
    tr_data_items = osp.join(data_root,f'split/{n_fold}_fold_tr_items.npy')
    val_data_items = osp.join(data_root,f'split/{n_fold}_fold_val_items.npy')  
    test_data_items = osp.join(data_root,f'split/{n_fold}_fold_test_items.npy')
    
    tr_dataset = MyInMemoryDataset(data_root,tr_data_items,celllines_data,drugs_data,args=args)
    val_dataset = MyInMemoryDataset(data_root,val_data_items,celllines_data,drugs_data,args=args)
    test_dataset = MyInMemoryDataset(data_root,test_data_items,celllines_data,drugs_data,args=args)
    """
    tr_data_items = osp.join(data_root, f'split/{n_fold}_fold_tr_item_dose_matched.npy')
    val_data_items = osp.join(data_root, f'split/{n_fold}_fold_val_items_dose_matched.npy')
    test_data_items = osp.join(data_root, f'split/{n_fold}_fold_test_items_dose_matched.npy')
    
    # 加载训练数据并限制为前10000条
    tr_items_raw = np.load(tr_data_items, allow_pickle=True)
    tr_items_small = tr_items_raw[:7000]
    temp_tr_path = osp.join(data_root, f'split/0_temp_tr_items.npy')
    np.save(temp_tr_path, tr_items_small)

    # 用临时训练数据构建数据集
    tr_dataset = MyInMemoryDataset(data_root, temp_tr_path, celllines_data, drugs_data, args=args,force_reprocess=True)
    val_dataset = MyInMemoryDataset(data_root, val_data_items, celllines_data, drugs_data, args=args)
    test_dataset = MyInMemoryDataset(data_root, test_data_items, celllines_data, drugs_data, args=args)

    tr_dataloader = DataLoader(tr_dataset[:7000], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset[:1600], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset[:1780], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    '''# 不再截断，直接用完整数据集
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)'''

    
    print(f'train data:{len(tr_dataloader)*args.batch_size}')  
    print(f'Valid data:{len(val_dataloader)*args.batch_size}')
    print(f'Test data:{len(test_dataloader)*args.batch_size}')
    
    return tr_dataloader, val_dataloader, test_dataloader



def load_dataloader_with_dose(n_fold, args):
    work_dir = args.workdir
    data_root = osp.join(work_dir, 'data')

    # 加载细胞特征
    if args.celldataset == 1:
        celllines_data = osp.join(data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root, '/root/lanyun-tmp/Project/SynergyX/data/0_cell_data/4079g/4cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes.npy')

    drugs_data = osp.join(data_root, '/root/lanyun-tmp/Project/SynergyX/data/1_drug_data/4drugSmile_drugSubEmbed_canonical.npy')

    tr_data_items = osp.join(data_root, f'split/{n_fold}_fold_tr_item_dose_matched.npy')
    val_data_items = osp.join(data_root, f'split/{n_fold}_fold_val_items_dose_matched.npy')
    test_data_items = osp.join(data_root, f'split/{n_fold}_fold_test_items_dose_matched.npy')
    
    # 加载训练数据并限制为前10000条
    tr_items_raw = np.load(tr_data_items, allow_pickle=True)
    tr_items_small = tr_items_raw[:7000]
    temp_tr_path = osp.join(data_root, f'split/0_temp_tr_items.npy')
    np.save(temp_tr_path, tr_items_small)

    # 用临时训练数据构建数据集
    tr_dataset = MyInMemoryDataset(data_root, temp_tr_path, celllines_data, drugs_data, args=args,force_reprocess=True)
    val_dataset = MyInMemoryDataset(data_root, val_data_items, celllines_data, drugs_data, args=args)
    test_dataset = MyInMemoryDataset(data_root, test_data_items, celllines_data, drugs_data, args=args)

    tr_dataloader = DataLoader(tr_dataset[:7000], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset[:1600], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset[:1780], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)


    print(f'train data:{len(tr_dataloader)*args.batch_size}')  
    print(f'Valid data:{len(val_dataloader)*args.batch_size}')
    print(f'Test data:{len(test_dataloader)*args.batch_size}')
    
    return tr_dataloader, val_dataloader, test_dataloader

def load_infer_dataloader_with_dose(args):
    data_root = osp.join(args.workdir, 'data')
    if args.celldataset == 1:
        celllines_data = osp.join(data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root, '/root/lanyun-tmp/Project/SynergyX/data/0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes.npy')
    
    drugs_data = osp.join(data_root,  '/root/lanyun-tmp/Project/SynergyX/data/1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    data_items = args.infer_path
    # 确保 MyInMemoryDataset 里的 __getitem__ 返回 doseA/doseB
    infer_dataset = MyInMemoryDataset(data_root, data_items, celllines_data, drugs_data, args=args, include_dose=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4)
    infer_data_arr = np.load(data_items, allow_pickle=True)

    return infer_dataloader, infer_data_arr



def train_with_dose(model, criterion, opt, dataloader, device, args=None):
    model.train()
    train_loss_sum = 0
    i = 0
    lr_list = []

    pbar = tqdm(dataloader, desc=f"Training", unit="batch")
    for data in pbar:
        i += 1
        model.zero_grad()
        data = data.to(device)

        drugA = data.drugA
        drugB = data.drugB
        x_cell = data.x_cell
        doseA = data.doseA
        doseB = data.doseB
        label = data.y.unsqueeze(1).float()

        output, _, _ = model(drugA, drugB, x_cell, doseA, doseB)
        train_loss = criterion(output, label)
        train_loss_sum += train_loss.item()
        train_loss.backward()
        opt.step()
        pbar.set_postfix({"Batch Loss": train_loss.item()})

    avg_loss = train_loss_sum / i
    return avg_loss, lr_list



        
 

# def train(model,criterion,opt,dataloader,device,args=None,lr_scheduler=None):

#     model.train()
#     train_loss_sum = 0
#     i = 0 
#     lr_list = []

#     for data in dataloader:
#         i += 1
#         model.zero_grad()
#         x = data.to(device)
#         output, _ = model(x)
        
#         if args.task == 'reg':
#             y = data.y.unsqueeze(1).type(torch.float32).to(device)
#             # y = Scalr(y)
#             train_loss = criterion(output,y)
#         if args.task == 'clf':
#             y = data.y_clf.long().to(device)
#             train_loss = criterion(output,y)

#         train_loss_sum += train_loss
#         # opt.optimizer.zero_grad()
#         # opt.zero_grad()
#         train_loss.backward() 

#         if lr_scheduler == 0:           
#             lr = opt.step()
#             # print(f'{i} batch lr: {lr}') 
#             lr_list.append(lr.cpu().detach().item())
#         else:
#             opt.step()
#             lr = opt.param_groups[0]['lr']
#             lr_list.append(lr)  # 返回实时的学习率

#     train_loss_sum = train_loss_sum.cpu().detach().numpy()       
#     loss = train_loss_sum/i
    
#     return loss,lr_list



def validate_with_dose(model, criterion, dataloader, device, args=None):
    model.eval()
    y_true = []
    y_pred = []
    i = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating", unit="batch"):
            i += 1
            data = data.to(device)
            drugA = data.drugA
            drugB = data.drugB
            x_cell = data.x_cell
            doseA = data.doseA
            doseB = data.doseB
            label = data.y.unsqueeze(1).float()

            output, _, _ = model(drugA, drugB, x_cell, doseA, doseB)
            y_true.append(label)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    mse, rmse, mae, r2, pearson, spearman = get_metrics(y_true, y_pred)
    return mse, rmse, mae, r2, pearson, spearman, None





from tqdm import tqdm

def infer_with_dose(model, dataloader, device, args=None):
    model.eval()
    y_pred = []
    cell_embed = []
    attn = []
    i = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Inferencing", unit="batch"):
            i += 1
            data = data.to(device)
            drugA = data.drugA
            drugB = data.drugB
            x_cell = data.x_cell
            doseA = data.doseA
            doseB = data.doseB

            output, output_cell_embed, output_attn = model(drugA, drugB, x_cell, doseA, doseB)
            if args.output_attn:
                cell_embed.append(output_cell_embed)
                attn.append(output_attn)
            y_pred.append(output)

    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    cell_embed = torch.stack(cell_embed, dim=0).cpu().detach().numpy() if cell_embed else None
    attn = torch.stack(attn, dim=0).cpu().detach().numpy() if attn else None

    return y_pred, cell_embed, attn
