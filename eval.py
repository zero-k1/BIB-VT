from torch.utils.data import DataLoader
from bib_data.datasets import FrameDataset
import torch
from bib_vt.transformer import VideoTransformer
import torch.nn as nn
from bib_vt.utils import pred_step, collate, remove_padding, get_agent_steps
from bib_vt.settings import NR_EPS, IMG_SZ_COMPRESSED, IMG_SZ, \
    DIM, DIM_HEAD, DEPTH, HEADS, MLP_DIM, device, data_path_eval, BATCH_SIZE, MAX_LENGTH, NR_GPUs, NR_WORKERS, \
    data_path_train, model_dir
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = str([x for x in range(NR_GPUs)])[1:-1]

def init_dataloaders(bib_task, test_task):
    dataset_eval = None if bib_task is None else FrameDataset(data_path_eval, types=[bib_task],
                                mode='eval', process_data=0, device=device)
    dataloader_eval = None if bib_task is None else DataLoader(dataset=dataset_eval, batch_size=2, num_workers=0, shuffle=False,
                                 drop_last=True, collate_fn=collate)
    dataset_test = None if test_task is None else FrameDataset(data_path_train,
                               types=[test_task],
                               mode='test', process_data=0, device=device)
    dataloader_test = None if test_task is None else DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, num_workers=NR_WORKERS, shuffle=True,
                                drop_last=True, collate_fn=collate)
    return dataloader_eval if bib_task is not None else dataloader_test

def get_z_scores(total, total_expected, total_unexpected):
    mean = torch.mean(torch.Tensor(total))
    std = torch.std(torch.Tensor(total))
    print("Z-Score expected: ",
          (torch.mean(torch.Tensor(total_expected)) - mean) / std)
    print("Z-Score unexpected: ",
          (torch.mean(torch.Tensor(total_unexpected)) - mean) / std)

def eval(model, iterator, bce, mse, weighting):
    model.eval()
    epoch_loss = 0
    for i, (agent_pos, padding, frames) in enumerate(iterator):
        agent_pos = agent_pos.to(device)
        frames = frames.to(device)
        for ep in range(NR_EPS - 1, NR_EPS):
            with torch.no_grad():
                loss = pred_step(model, frames, padding, ep, agent_pos, weighting, mse, bce)
            epoch_loss += float(loss)
    return epoch_loss / len(iterator)


def evaluate_bib(model, iterator, bce):
    correct, incorrect = 0, 0
    total, total_expected, total_unexpected = [], [], []
    for i, (agent_pos, padding, frames) in enumerate(iterator):
        frames = frames.to(device)
        padding = padding.to(device)
        agent_pos = agent_pos.to(device)
        for ep in range(NR_EPS - 1, NR_EPS):
            model.eval()
            agent_steps = get_agent_steps(agent_pos[:, ep, :])
            with torch.no_grad():
                agent_pos_pred, image_recon = model(frames[:, ep], padding, frames[:, :ep], MAX_LENGTH)
            agent_pos_pred, agent_steps, ep_frames, image_recon = remove_padding(padding, ep, agent_pos_pred,
                                                                                 agent_steps, frames, image_recon)
            ep_lengths = padding[:, ep, :].sum(dim=-1)
            loss_e_bce = bce(agent_pos_pred[:ep_lengths[0]], agent_steps[:ep_lengths[0]]).sum(-1).sum(-1).sum(-1)
            expected = loss_e_bce.max()
            loss_ue_bce = bce(agent_pos_pred[ep_lengths[0]:], agent_steps[ep_lengths[0]:]).sum(-1).sum(-1).sum(-1)
            unexpected = loss_ue_bce.max()
            total_expected.append(expected)
            total_unexpected.append(unexpected)
            total.append(unexpected)
            total.append(expected)
            if expected < unexpected:
                correct += 1
            else:
                incorrect += 1
    print(correct / (correct + incorrect))
    get_z_scores(total, total_expected, total_unexpected)

def main():
    MODEL_CHECKPOINT = model_dir + sys.argv[3]
    vt = nn.DataParallel(
        VideoTransformer(dim=DIM, depth=DEPTH, dim_head=DIM_HEAD, heads=HEADS, mlp_dim=MLP_DIM, device=device,
                         batch_size=BATCH_SIZE)).to(device)
    vt.load_state_dict(torch.load(MODEL_CHECKPOINT))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([IMG_SZ_COMPRESSED * 2]).to(device))
    weighting = IMG_SZ ** 2 / IMG_SZ_COMPRESSED ** 2
    mse = nn.MSELoss()
    if sys.argv[1] == "eval":
        dataloader = init_dataloaders(sys.argv[2], None)
        evaluate_bib(vt, dataloader, bce)
    elif sys.argv[1] == "test":
        dataloader = init_dataloaders(None, sys.argv[2])
        eval(vt, dataloader, bce, mse, weighting)
    else:
        print("First argument should be 'eval' or 'test'")

if __name__ == "__main__":
    main()