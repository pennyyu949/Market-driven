import numpy as np

def california_reward(pred, true, over, under, over_higher, under_lower, ic, dc):
    losses = []
    pred = np.asarray(pred).flatten() 
    true = np.asarray(true).flatten()
    for p, t in zip(pred, true):
        diff = p - t 
        if diff >= 0:
            if diff <= 0.015 * t:
                loss = 0.015 * ic
            elif 0.015 * t < diff <= 0.075 * t:
                loss = 0.015 * ic + (diff - 0.015 * t) * over * ic
            else:
                loss = 0.015 * ic + 0.06 * t* over * ic + (diff - 0.075 * t) * over_higher * ic
        else:
            diff = np.abs(diff)
            if diff <= 0.015 * t:
                loss = 0.015 * dc
            elif 0.015 * t < diff <= 0.075 * t:
                loss = 0.015 * dc + (diff - 0.015 * t) * under * dc
            else:
                loss = 0.015 * dc + 0.06 * under * t* dc + (diff - 0.075 * t) * under_lower * dc
        losses.append(loss)

    losses = [l for l in losses if not np.isnan(l)]
    return np.mean(losses)

def catalonia_reward(pred, true, spanish_downward, spanish_upward, unit):
    # downward means true is less than pred
    # upward means production is more than forecasted
    if unit == 'Wh':
        # Wh to MWh
        pred = pred/ 1000000
        true = true/ 1000000
    elif unit == 'kWh':
        # kWh to MWh
        pred = pred/ 1000
        true = true/ 1000
        
    losses = []
    for p, t in zip(pred, true):
        
        condition = p >= t
        loss_over = spanish_downward * (p - t) 
        loss_under = spanish_upward * (t - p) 
        loss = np.where(condition, loss_over, loss_under)
        # losses.append(np.sum(loss))   
        losses.append(loss)
    return np.mean(losses)

def victoria_reward(pred, true, under, over):
    condition = pred >= true
    loss = np.where(condition, over * (pred - true), under * (true - pred))
    return np.mean(loss)

def victoria_reward_per_instance(pred, true, under, over):
    condition = pred >= true
    return np.where(condition, over * (pred - true), under * (true - pred))
    
def MSE(pred, true):
    return np.mean((pred - true) ** 2) 

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
