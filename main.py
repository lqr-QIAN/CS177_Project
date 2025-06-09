import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np

from models.model import SynergyxNet
from utlis import (EarlyStopping, collect_env, load_dataloader_with_dose, load_infer_dataloader_with_dose, 
                   set_random_seed, train_with_dose, validate_with_dose, infer_with_dose)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='patience for earlystopping')
    parser.add_argument('--resume-from', type=str, help='the path of pretrained_model')
    parser.add_argument('--mode', type=str, default='train', help='train, test or infer')               
    parser.add_argument('--omic', type=str, default='exp,mut,cn', help="omics_data")
    parser.add_argument('--workdir', type=str, default='/root/lanyun-tmp/Project/SynergyX', help='working directory')
    parser.add_argument('--celldataset', type=int, default=2, help='gene set')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans', help='cell encoder type')
    parser.add_argument('--nfold', type=str, default='0', help='dataset fold')
    parser.add_argument('--saved-model', type=str, default='./saved_model/0_fold_SynergyX.pth', help='trained model path')
    parser.add_argument('--infer-path', type=str, default='./data/infer_data/sample_infer_items.npy', help='infer data path')
    parser.add_argument('--output-attn', type=int, default=0, help="output attention/cell embedding")
    parser.add_argument('--use-geneformer', type=int, default=0, help='Geneformer flag')
    parser.add_argument('--geneformer-hidden-size', type=int, default=256, help='Geneformer size')
    parser.add_argument('--use-scgpt', type=int, default=0, help='scGPT flag')
    parser.add_argument('--scgpt-hidden-size', type=int, default=512, help='scGPT hidden size')
    parser.add_argument('--scgpt-dropout', type=float, default=0.1, help='scGPT dropout')
    return parser.parse_args()

def main():
    args = arg_parse()
    set_random_seed(args.seed)
    device = args.device

    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    expt_folder = osp.join('experiment/', f'{timestamp}')
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()]) 
    dash_line = '-' * 60 + '\n'  
    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    print('\n--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('\n')
    
    if args.mode == 'train':
        nfold = [i for i in args.nfold.split(',')]
        for k in nfold:
            model = SynergyxNet(args=args).to(device)
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.2fM" % (total/1e6))
            model.init_weights()
            criterion = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            start_epoch = 0

            if args.resume_from:
                resume_path = args.resume_from
                pretrain_dict = torch.load(resume_path)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                start_epoch = int(osp.basename(resume_path).split('_')[0]) + 1
                print(f'Load pre-trained parameters successfully! From epoch {start_epoch} to train……')

            tr_dataloader, val_dataloader, test_dataloader = load_dataloader_with_dose(n_fold=k, args=args)

            start_time = time.time()
            print(f'{k}_Fold_Training is starting. Start_time:{timestamp}')

            stopper = EarlyStopping(mode='lower', metric='mse', patience=args.patience, n_fold=k, folder=expt_folder, min_delta=1.0) 
            for epoch in range(start_epoch, args.epochs):
                train_loss, lr_list = train_with_dose(model=model, criterion=criterion, opt=optimizer, dataloader=tr_dataloader, device=device, args=args)
                val_loss, _, _, _, _, _, _ = validate_with_dose(model=model, criterion=criterion, dataloader=val_dataloader, device=device, args=args)
                print('Epoch %d, Train_loss %f, Valid_loss %f' % (epoch, train_loss, val_loss)) 
                if stopper.step(val_loss, model):
                    print('EarlyStopping! Finish training!')
                    break 

            print(f'{k}_fold training is done! Training_time:{(time.time() - start_time)/60}min')
            print('Start testing ... ')

            stopper.load_checkpoint(model)
            m1, m2, m3, m4, m5, m6, m7 = validate_with_dose(model=model, criterion=criterion, dataloader=tr_dataloader, device=device, args=args)
            n1, n2, n3, n4, n5, n6, n7 = validate_with_dose(model=model, criterion=criterion, dataloader=val_dataloader, device=device, args=args)
            print('Train result: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f} '.format(m1, m2, m3, m4, m5, m6))
            print('Val result: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f}'.format(n1, n2, n3, n4, n5, n6))
        print('All folds training is completed!')

    elif args.mode == 'test':
        print('Test mode:')
        model = SynergyxNet(args=args).to(device)
        model.load_state_dict(torch.load(args.saved_model))
        criterion = torch.nn.MSELoss(reduction='mean')

        k = osp.basename(args.saved_model).split('_')[0]
        tr_dataloader, val_dataloader, test_dataloader = load_dataloader_with_dose(n_fold=k, args=args)
        l1, l2, l3, l4, l5, l6, _ = validate_with_dose(model=model, criterion=criterion, dataloader=test_dataloader, device=device, args=args)
        print('Test result: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f}\n'.format(l1, l2, l3, l4, l5, l6))

    elif args.mode == 'infer':
        print('Infer mode:')
        model = SynergyxNet(args=args).to(device)
        model.load_state_dict(torch.load(args.saved_model))
        infer_dataloader, infer_data_arr = load_infer_dataloader_with_dose(args=args)
        y_pred_arr, cell_embed_arr, attn_arr = infer_with_dose(model=model, dataloader=infer_dataloader, device=device, args=args)
        output_arr = np.concatenate((infer_data_arr, y_pred_arr), axis=1)

        print('Inference done! Saving to file……')
        output_df = pd.DataFrame(output_arr, columns=['drugA', 'drugB', 'sample_id', 'label', 'S_pred'])
        output_df.to_csv(f'experiment/{timestamp}/predict_res.csv')

        if args.output_attn:
            np.save(f'experiment/{timestamp}/cell_embed.npy', cell_embed_arr)
            np.save(f'experiment/{timestamp}/attn.npy', attn_arr)

if __name__ == '__main__':
    main()
