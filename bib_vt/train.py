from torch.utils.data import DataLoader
from bib_data.datasets import FrameDataset
import torch
from bib_vt.transformer import VideoTransformer
import torch.nn as nn
from torch import optim
import numpy as np
import os
from bib_vt.utils import pred_step, collate
from bib_vt.eval import eval
from tqdm import tqdm
from bib_vt.settings import BATCH_SIZE, NR_EPS, IMG_SZ, IMG_SZ_COMPRESSED, NR_GPUs, NR_WORKERS, \
    DIM, DIM_HEAD, DEPTH, HEADS, MLP_DIM, NR_EPOCHS, VAL_INTERVAL, device, data_path_train, model_dir, model_name

np.set_printoptions(precision=5)
os.environ['CUDA_VISIBLE_DEVICES'] = str([x for x in range(NR_GPUs)])[1:-1]

def init_dataloaders():
    dataset_train = FrameDataset(data_path_train,
                                 types=['single_object', 'preference', 'multi_agent', 'instrumental_action'],
                                 mode='train', process_data=0, device=device)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, num_workers=NR_WORKERS, shuffle=True,
                                  drop_last=True, collate_fn=collate, pin_memory=True)

    dataset_val = FrameDataset(data_path_train,
                               types=['single_object', 'preference', 'multi_agent', 'instrumental_action'],
                               mode='val', process_data=0, device=device)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, num_workers=NR_WORKERS, shuffle=True,
                                drop_last=True, collate_fn=collate)
    return dataloader_train, dataloader_val

def val_step(model, best_valid_loss, dataloader_val, mse, bce, weighting):
    val_loss = eval(model, dataloader_val, bce, mse, weighting)
    print(val_loss)
    if val_loss < best_valid_loss:
        print("Saving!")
        torch.save(model.state_dict(), model_dir + model_name)
        best_valid_loss = val_loss
    return best_valid_loss


def train(model, iterator, optimizer, mse, bce, best_valid_loss, weighting, dataloader_val, epoch):
    model.train()
    epoch_loss = 0
    with tqdm(iterator, unit="batch") as tepoch:
        for i, (agent_pos, padding, frames) in enumerate(tepoch):  #
            tepoch.set_description(f"Epoch {epoch}")
            agent_pos = agent_pos.to(device)
            frames = frames.to(device)
            for ep in range(1, NR_EPS):
                optimizer.zero_grad()
                loss = pred_step(model, frames, padding, ep, agent_pos, weighting, mse, bce)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            epoch_loss += float(loss)
            if i % VAL_INTERVAL == VAL_INTERVAL - 1:
                best_valid_loss = val_step(model, best_valid_loss, dataloader_val, mse, bce, weighting)
                model.train()
    return epoch_loss / len(iterator), best_valid_loss


def main():
    dataloader_train, dataloader_val = init_dataloaders()
    best_valid_loss = float('inf')
    vt = nn.DataParallel(
        VideoTransformer(dim=DIM, depth=DEPTH, dim_head=DIM_HEAD, heads=HEADS, mlp_dim=MLP_DIM, device=device,
                         batch_size=BATCH_SIZE)).to(device)
    optimizer = optim.Adamax(vt.parameters())
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([IMG_SZ_COMPRESSED * 2]).to(device))
    weighting = IMG_SZ ** 2 / IMG_SZ_COMPRESSED ** 2
    for epoch in range(NR_EPOCHS):
        train_loss, best_valid_loss = train(vt, dataloader_train, optimizer, mse, bce,
                                                        best_valid_loss, weighting, dataloader_val, epoch)
        best_valid_loss = val_step(vt, best_valid_loss, dataloader_val, mse, bce, weighting)
        print(epoch, f'\tTrain Loss: {train_loss:.3f}')

if __name__ == "__main__":
    main()
