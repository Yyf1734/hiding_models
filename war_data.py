from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from warfit_learn import datasets, preprocessing




class WarfitDataset(Dataset):
    def __init__(self, df, target_column='Therapeutic Dose of Warfarin'):
        self.ys = df[target_column].values.astype(np.float32)
        self.Xs = df.drop([target_column], axis=1).values.astype(np.float32)

    def __getitem__(self, index):
        X, y = self.Xs[index], self.ys[index]
        return X, y
        
    def __len__(self):
        return self.Xs.shape[0]
    
    
def get_warfit_dataloader(BATCH_SIZE=32):
    raw_iwpc = datasets.load_iwpc()
    df = preprocessing.prepare_iwpc(raw_iwpc)
    df['Height (cm)'] = df['Height (cm)'] / 100
    df['Weight (kg)'] = df['Weight (kg)'] / 100
    df = shuffle(df)
    
    pos = int(len(df) * 0.1)
    train_df = df[0 : 7 * pos]
    val_df = df[7 * pos : 8 * pos]
    test_df = df[8 * pos :]
    print(len(train_df), len(val_df), len(test_df))
    
    train_dataset = WarfitDataset(train_df)
    val_dataset = WarfitDataset(val_df)
    test_dataset = WarfitDataset(test_df)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, test_loader


class LR_Flexible(nn.Module):
    def __init__(self, layers:list):
        super(LR_Flexible, self).__init__()
        assert len(layers) >= 1
        self.layer_num = len(layers)
        
        hidden_layers = []
        for i in range(self.layer_num):
            if i + 1 < self.layer_num:
                hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
                hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(layers[self.layer_num - 1], 1)

    def forward(self, x):
        output = x
        for i in range(len(self.hidden_layers)):
            output = self.hidden_layers[i](output)

        return self.output_layer(output)
    
