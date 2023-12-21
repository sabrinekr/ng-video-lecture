"""
Trains a GPT to predict the next characters.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import ml_collections

from gpt import GPTLanguageModel

class NextCharDataset(Dataset):
    """
    Creates dataset to predict next char.
    """
    def __init__(self, config, split):
        self.config = config
        self.split = split

        input_file = self.config.input_file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] 

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        self.data = data[n:] if split == "test" else data[:n]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        block_size = self.config.block_size
        x = self.data[idx:idx+block_size] 
        y = self.data[idx+1:idx+block_size+1]
        return x, y

def get_config():
    config = ml_collections.ConfigDict()
    config.input_file = 'input.txt'
    config.batch_size = 64
    config.block_size = 256
    config.max_iters = 5000
    config.eval_interval = 500
    config.learning_rate = 3e-4
    config.eval_iters = 200
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

def collate_fn(batch):
  
  (xx, yy) = zip(*batch)

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy_pad


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()

    # construct train and test datasets
    train_dataset = NextCharDataset(config, split='train')
    test_dataset  = NextCharDataset(config, split='test')

    model = GPTLanguageModel()
    m = model.to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=0, drop_last=False)
    test_data_iter = iter(test_loader)
    data_iter = iter(train_loader)
    iter = 0

    while True:

        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = [t.to(config.device) for t in batch]
        x, y = batch

        # forward the model
        logits, loss = model(x, y)
        print("loss", loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iter += 1

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            out = {}
            with torch.no_grad():
                model.eval()
                losses = torch.zeros(config.eval_iters)
                for k in range(config.eval_iters):
                    batch_val = next(test_data_iter)
                    batch_val = [t.to(config.device) for t in batch_val]
                    x, y = batch_val
                    _, loss = model(x, y)
                    losses[k] = loss.item()
                out["val"] = losses.mean()
                model.train()
                print(f"step {iter}: val loss {losses['val']:.4f}")
        # termination conditions
        if config.max_iters is not None and iter >= config.max_iters:
            break




