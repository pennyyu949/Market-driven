import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import StandardScaler

def create_sequence(args, flag = 'train'):
    input_data = []
    output_data = []
    scaler = StandardScaler()
    attr_power = args.target
    sequence_length = args.seq_len
    forecast_step = args.pred_len
    if flag == 'train':
        data = pd.read_csv(os.path.join(args.root_path, 'train.csv'))
        scaler.fit(data[attr_power].values.reshape(-1, 1))
        if args.inverse:
            data[attr_power] = scaler.transform(data[attr_power].values.reshape(-1, 1))

    else:
        data = pd.read_csv(os.path.join(args.root_path, 'test.csv'))
        if args.inverse:
            train = pd.read_csv(os.path.join(args.root_path, 'train.csv'))
            scaler.fit(train[attr_power].values.reshape(-1, 1))
            data[attr_power] = scaler.transform(data[attr_power].values.reshape(-1, 1))
        
    data[args.time_feature] = pd.to_datetime(data[args.time_feature])

    power = data[attr_power].values
    
    for i in range(len(data) - sequence_length - (forecast_step - 1)):
        if np.isnan(power[i:i+sequence_length+forecast_step]).any():
            continue
        else:
            # Extract the input sequence
            input_seq = np.array([power[i:i+sequence_length], data.iloc[i:i+sequence_length][args.time_feature]])
            input_data.append(input_seq.T)

            # Extract the output value for prediction
            output_value = np.array([power[i+sequence_length:i+sequence_length+forecast_step], data.iloc[i+sequence_length:i+sequence_length+forecast_step][args.time_feature]])
            output_data.append(output_value.T)

    return np.array(input_data), np.array(output_data), scaler
