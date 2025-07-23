from xml.parsers.expat import model
from data_provider.data_generator import create_sequence
from models import GRU, MLP, LSTM, Crossformer
from data_provider.dataset import SolarDataset
from torch.utils.data import DataLoader
import torch
import os
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.custom_loss import CustomLoss
import wandb 
from utils.metrics import RMSE, victoria_reward, california_reward, catalonia_reward

class Exp_Validation:
    def __init__(self, args, custom, fix_seed = 2024):
        self.args = args
        self.custom = custom
        self.fix_seed = fix_seed
        self.device = self._acquire_device()
        self.model_dict = {'GRU': GRU, 
                           'MLP': MLP, 
                           'LSTM': LSTM, 
                           'Crossformer': Crossformer }
        self.model = self._build_model().to(self.device)
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _select_criterion(self):
        if not self.custom:
            criterion = nn.MSELoss()
        else:
            criterion = CustomLoss(over_penalty=self.args.over_penalty, under_penalty=self.args.under_penalty)
        return criterion
        
    
    def test(self, model):
        
        input_data, output_data, scaler = create_sequence(self.args, flag='test')
        input_data = input_data[:,:,:-1].reshape(input_data.shape[0], -1)
        output_power = output_data[:, :, -2]
        input_data = input_data.astype(np.float32)
        output_power = output_power.astype(np.float32)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_power = torch.tensor(output_power, dtype=torch.float32)
        output_time = output_data[:, 0, -1]
        output_time = pd.to_datetime(output_time)
        input_data = input_data.unsqueeze(2)    
        output_power = output_power.unsqueeze(2)
        indices = list(range(len(output_power)))
        test_ds = SolarDataset(input_data, output_power, indices)
        test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        weights = torch.load(model)  
        self.model.load_state_dict(weights)     
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
             
                outputs = self.model(batch_x, dec_inp)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if self.args.inverse:
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    batch_y = batch_y.reshape(-1, batch_y.shape[-1])
                    outputs = scaler.inverse_transform(outputs)
                    batch_y = scaler.inverse_transform(batch_y)
                    outputs = outputs.reshape(-1, self.args.pred_len, outputs.shape[-1])
                    batch_y = batch_y.reshape(-1, self.args.pred_len, batch_y.shape[-1])

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)        
        
        preds = np.concatenate([pred for pred in preds], axis=0)
        trues = np.concatenate([true for true in trues], axis=0)

        rmse = RMSE(preds, trues)
        custom_metric = -999
        if self.args.data == 'victoria': 
            custom_metric = victoria_reward(preds, trues, self.args.under_penalty, self.args.over_penalty)
            return rmse, custom_metric, preds, trues, output_time
        elif self.args.data == 'california':
            # due to the requirements from california regulation
            preds = preds[:,-1, :]
            trues = trues[:,-1, :]
            print(f'Preds shape: {preds.shape}, Trues shape: {trues.shape}')
            
            custom_metric = california_reward(preds, trues, self.args.over_penalty, self.args.under_penalty, self.args.overhigher, self.args.underlower, self.args.icdc, self.args.icdc)
            return rmse, custom_metric, preds, trues, output_time
        elif self.args.data == 'catalonia':
            score_catalonia = {}
            for i in range(preds.shape[-2]):
                score_catalonia[i] = {}
                df = pd.DataFrame({'output_time': pd.to_datetime(output_time) + pd.DateOffset(hours=i), 'preds': preds[:, i, :].reshape(-1), 'trues': trues[:, i, :].reshape(-1)})
                for date, group_data in df.groupby(df['output_time'].dt.date):
                    score = catalonia_reward(group_data['preds'], group_data['trues'], self.args.spanish_downwards, self.args.spanish_upwards, self.args.capacity)
                    score_catalonia[i][date] = score

            custom_metric = pd.DataFrame(score_catalonia).dropna().mean().mean()
            preds = preds.reshape(preds.shape[0], -1)
            trues = trues.reshape(trues.shape[0], -1)

            return rmse, custom_metric, preds, trues, output_time
