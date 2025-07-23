import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
import wandb
from exp.exp_validation import Exp_Validation

if __name__ == '__main__':
    fix_seed = 2024
    os.environ['PYTHONHASHSEED'] = str(fix_seed)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='MarketDrivenForecast')
    parser.add_argument('--model', type=str, default='Crossformer',
                        help='model name, options: [GRU, MLP, LSTM, Crossformer]')
    
    parser.add_argument('--data', type=str, default='victoria', help='dataset type, options: [victoria, california]')
    parser.add_argument('--root_path', type=str, default='./data/victoria/', help='root path of the data file')
    parser.add_argument('--mse_model_path', type=str, default='checkpoints/victoria/modelMLP_seqlen4_labellen2_predlen1_over1.2_under1.0/checkpoint_mse.pth', help='path to the MSE model')
    parser.add_argument('--custom_model_path', type=str, default='checkpoints/victoria/modelMLP_seqlen4_labellen2_predlen1_over1.2_under1.0/checkpoint_custom.pth', help='path to the Custom model')
    parser.add_argument('--time_feature', type=str, default='date', help='time feature')
    parser.add_argument('--target', type=str, default='obs_power', help='target variable')
    
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=2, help='start token length')
    parser.add_argument('--feature_size', type=int, default=1, help='feature size of the input data, if single feature data, set to 1')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--layer_dim', type=int, default=1, help='layer dim of the model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
    parser.add_argument('--inverse', type=int, default=1, help='inverse output data')
    parser.add_argument('--factor', type=int, default=1, help='factor of the model')
    parser.add_argument('--d_ff', type=int, default=2048, help='d_ff of the model')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads of the model')
    
    parser.add_argument('--over_penalty', type=float, default=1.2, help='penalty for over forecast')
    parser.add_argument('--under_penalty', type=float, default=1.0, help='penalty for under forecast')
    parser.add_argument('--spanish_downwards', type=float, default=32.82, help='spanish downwards')
    parser.add_argument('--spanish_upwards', type=float, default=56.67, help='spanish upwards')
    parser.add_argument('--icdc', type=float, default=0.111, help='us base cost')
    parser.add_argument('--overhigher', type=float, default=1.25, help='us extra over penalty')
    parser.add_argument('--underlower', type=float, default=0.75, help='us extra under penalty')
    parser.add_argument('--capacity', type=str, default='Wh', help='capacity for the power station')
    
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    
    wandb.init(project='MarketDrivenForecast', reinit=True)
    wandb.config.update(vars(args), allow_val_change=True)
        
    exp_mse = Exp_Validation(args, custom=False, fix_seed = fix_seed)
    exp_custom = Exp_Validation(args, custom=True, fix_seed = fix_seed)
            
    model_mse = args.mse_model_path
    model_custom = args.custom_model_path
    
    rmse_mse, custom_mse, preds_mse, _, _ = exp_mse.test(model_mse)
    rmse_custom, custom_custom, preds_custom, _ , _ = exp_custom.test(model_custom)
        
    wandb.log({'rmse_mse': rmse_mse, 'rmse_custom': rmse_custom, 'custom_metric_mseloss': custom_mse, 'custom_metric_customloss': custom_custom})
    print(f'RMSE for MSELoss: {rmse_mse}, RMSE for CustomLoss: {rmse_custom}')
    print(f'Custom for MSELoss: {custom_mse}, Custom for CustomLoss: {custom_custom}')

    f = open(f"result_{args.data}_{args.model}_{args.over_penalty}_{args.under_penalty}_forecast.txt", 'a')
    f.write(f'RMSE for MSELoss: {rmse_mse}, RMSE for CustomLoss: {rmse_custom} \n')
    f.write(f'Custom for MSELoss: {custom_mse}, Custom for CustomLoss: {custom_custom} \n')
    f.close()