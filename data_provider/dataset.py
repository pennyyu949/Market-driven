from torch.utils.data import Dataset

class SolarDataset(Dataset):
    def __init__(self, time_features, power, indices):
        self.time_features = time_features
        self.power = power
        self.indices = indices
        
    def __len__(self):
        return len(self.power)
    
    def __getitem__(self, index):
        x = self.time_features[self.indices[index]]
        y = self.power[self.indices[index]]
        
        return x, y
