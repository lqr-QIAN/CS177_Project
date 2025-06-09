import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np
import gc

from models.model_scgpt import SynergyxNet
from utlis_scgpt import (EarlyStopping, collect_env, load_dataloader, load_infer_dataloader, 
                   set_random_seed, train, validate, infer)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=5, help='patience for earlystopping (default: 5)')
    parser.add_argument('--resume-from', type=str, help='the path of pretrained_model')
    parser.add_argument('--mode', type=str, default='train', help='train or test or infer')               
    parser.add_argument('--omic', type=str, default='exp,mut,cn', help="omics_data included in training")
    parser.add_argument('--workdir', type=str, default='/root/lanyun-tmp/Project/SynergyX', help='working directory')
    parser.add_argument('--celldataset', type=int, default=2, help='Using which geneset to train the model')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans', help='cell encoder')
    parser.add_argument('--nfold', type=str, default='0', help='dataset index')
    parser.add_argument('--saved-model', type=str, default='./saved_model/0_fold_SynergyX.pth', help='trained model path')  
    parser.add_argument('--infer-path', type=str, default='./data/infer_data/sample_infer_items.npy', help="infer data path")
    parser.add_argument('--output-attn', type=int, default=0, help="output attention matrix and cell embedding")
    parser.add_argument('--use-geneformer', type=int, default=0, help='whether to use Geneformer')
    parser.add_argument('--geneformer-hidden-size', type=int, default=256, help='Geneformer hidden size')
    parser.add_argument('--use-scgpt', type=int, default=1, help='whether to use scGPT')
    parser.add_argument('--scgpt-hidden-size', type=int, default=512, help='scGPT hidden size')
    parser.add_argument('--scgpt-dropout', type=float, default=0.1, help='scGPT dropout rate')
    return parser.parse_args()

def main():
    args = arg_parse()
    set_random_seed(args.seed)
    device = args.device

    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    expt_folder = osp.join('experiment/', f'{timestamp}')
    os.makedirs(expt_folder, exist_ok=True)

    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()]) 
    dash_line = '-' * 60 + '\n'  
    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    print('\n--------args----------')
    for k in vars(args):
        print(f'{k}: {getattr(args, k)}')
    print('\n')

    if args.mode == 'train':
        nfold = [i for i in args.nfold.split(',')]
        for k in nfold:
            model = SynergyxNet(args=args).to(device)
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.2fM" % (total/1e6))
            model.init_weights()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            start_epoch = 0

            if args.resume_from:
                resume_path = args.resume_from
                pretrain_dict = torch.load(resume_path, map_location=device)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                start_epoch = int(osp.basename(resume_path).split('_')[0]) + 1
                print(f'Loaded pre-trained parameters from {resume_path}, start from epoch {start_epoch}')

            tr_loader, val_loader, test_loader = load_dataloader(n_fold=k, args=args)
            stopper = EarlyStopping(mode='lower', metric='mse', patience=args.patience, n_fold=k, folder=expt_folder, min_delta=1.0)

            for epoch in range(start_epoch, args.epochs):
                train_loss, _ = train(model=model, criterion=criterion, opt=optimizer, dataloader=tr_loader, device=device, args=args)
                val_loss, *_ = validate(model=model, criterion=criterion, dataloader=val_loader, device=device, args=args)
                print(f'Epoch {epoch}, Train_loss: {train_loss:.4f}, Valid_loss: {val_loss:.4f}')

                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()

                if stopper.step(val_loss, model):
                    print('EarlyStopping! Training finished.')
                    break

            stopper.load_checkpoint(model)
            m1, m2, m3, m4, m5, m6, _ = validate(model=model, criterion=criterion, dataloader=tr_loader, device=device, args=args)
            n1, n2, n3, n4, n5, n6, _ = validate(model=model, criterion=criterion, dataloader=val_loader, device=device, args=args)
            print(f'Train: mse:{m1:.4f}, rmse:{m2:.4f}, mae:{m3:.4f}, r2:{m4:.4f}, pearson:{m5:.4f}, spearman:{m6:.4f}')
            print(f'Val: mse:{n1:.4f}, rmse:{n2:.4f}, mae:{n3:.4f}, r2:{n4:.4f}, pearson:{n5:.4f}, spearman:{n6:.4f}')

        print('All folds training completed!')

    elif args.mode == 'test':
        print('Test mode:')
        model = SynergyxNet(args=args).to(device)
        model.load_state_dict(torch.load(args.saved_model, map_location=device))
        tr_loader, val_loader, test_loader = load_dataloader(n_fold=osp.basename(args.saved_model).split('_')[0], args=args)
        criterion = torch.nn.MSELoss()
        l1, l2, l3, l4, l5, l6, _ = validate(model=model, criterion=criterion, dataloader=test_loader, device=device, args=args)
        print(f'Test: mse:{l1:.4f}, rmse:{l2:.4f}, mae:{l3:.4f}, r2:{l4:.4f}, pearson:{l5:.4f}, spearman:{l6:.4f}')

    elif args.mode == 'infer':
        print('Infer mode:')
        model = SynergyxNet(args=args).to(device)
        model.load_state_dict(torch.load(args.saved_model, map_location=device))
        infer_loader, infer_data_arr = load_infer_dataloader(args=args)

        output_file = f'{expt_folder}/predict_res.csv'
        with open(output_file, 'w') as f:
            f.write('drugA,drugB,sample_id,label,S_pred\n')

        for idx, data in enumerate(infer_loader):
            with torch.no_grad():
                output, cell_embed, attn = model(data.to(device))
                output = output.cpu().numpy()
                batch_info = infer_data_arr[idx * args.batch_size: (idx+1) * args.batch_size]
                batch_out = np.concatenate([batch_info, output], axis=1)
                pd.DataFrame(batch_out, columns=['drugA','drugB','sample_id','label','S_pred']).to_csv(output_file, mode='a', header=False, index=False)

                # Optional: Save embeddings & attn incrementally
                if args.output_attn:
                    np.save(f'{expt_folder}/cell_embed_batch_{idx}.npy', cell_embed.cpu().numpy())
                    np.save(f'{expt_folder}/attn_batch_{idx}.npy', attn.cpu().numpy())

            del data, output, cell_embed, attn
            gc.collect()
            torch.cuda.empty_cache()

        print('Inference completed, results saved.')

if __name__ == '__main__':
    main()
