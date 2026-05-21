import pandas as pd
import numpy as np
import mindspore.dataset as ds

class RatingDataset:
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.users = df['user_id'].values.astype(np.int32)
        self.items = df['item_id'].values.astype(np.int32)
        self.ratings = df['rating'].values.astype(np.float32)

    def __getitem__(self, index):
        return (self.users[index], self.items[index]), self.ratings[index]

    def __len__(self):
        return len(self.users)

def get_dataloader(data_path, batch_size=256, shuffle=True):
    dataset = ds.GeneratorDataset(
        source=RatingDataset(data_path),
        column_names=["ids", "rating"],
        shuffle=shuffle
    )
    dataset = dataset.batch(batch_size)
    return dataset