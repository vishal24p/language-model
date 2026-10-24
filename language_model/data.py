
import numpy as np
import torch


train_data = np.memmap("data/train.bin",dtype=np.uint16,mode="r",)
val_data = np.memmap("data/val.bin",dtype=np.uint16,mode="r",)


def get_batch(split, batch_size, context_length, device):

    data = train_data if split == "train" else val_data

    starts = np.random.randint(0,len(data) - context_length,size=batch_size,)



    x = np.stack([ data[i : i + context_length] for i in starts ])
    y = np.stack([ data[i + 1 : i + context_length + 1] for i in starts ])


    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)


    return x, y
