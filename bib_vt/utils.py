import torch
from bib_vt.settings import BATCH_SIZE, NR_EPS, MAX_LENGTH, IMG_SZ, IMG_CHANNELS, IMG_SZ_COMPRESSED, device

def get_agent_steps(ep_agent_pos):
    agent_steps = torch.zeros((BATCH_SIZE, MAX_LENGTH, IMG_SZ_COMPRESSED ** 2)).to(device)
    agent_steps.scatter_(dim=-1, index=ep_agent_pos.unsqueeze(-1),
                         src=torch.ones((BATCH_SIZE, MAX_LENGTH, 1)).to(device))
    agent_steps = agent_steps.reshape(BATCH_SIZE, MAX_LENGTH, IMG_SZ_COMPRESSED, IMG_SZ_COMPRESSED)
    return agent_steps


def remove_padding(padding, ep, agent_pos_pred, agent_steps, frames, image_recon):
    agent_pos_pred = agent_pos_pred.reshape(-1, IMG_SZ_COMPRESSED, IMG_SZ_COMPRESSED, 1)[
        padding[:, ep, :].reshape(-1)]
    agent_steps = agent_steps.reshape(-1, IMG_SZ_COMPRESSED, IMG_SZ_COMPRESSED, 1)[
        padding[:, ep, :].reshape(-1)]
    ep_frames = frames[:, ep, 1:].reshape(-1, IMG_CHANNELS, IMG_SZ, IMG_SZ)[
        padding[:, ep, :-1].reshape(-1)]
    image_recon = image_recon.reshape(-1, IMG_CHANNELS, IMG_SZ, IMG_SZ)[padding[:, ep, :].reshape(-1)]
    return agent_pos_pred, agent_steps, ep_frames, image_recon

def pred_step(model, frames, padding, ep, agent_pos, weighting, mse, bce):
    agent_pos_pred, image_recon = model(frames[:, ep], padding, frames[:, :ep], MAX_LENGTH)
    agent_steps = get_agent_steps(agent_pos[:, ep, :])
    agent_pos_pred, agent_steps, ep_frames, image_recon = remove_padding(padding, ep, agent_pos_pred,
                                                                         agent_steps, frames, image_recon)
    loss = bce(agent_pos_pred, agent_steps) + mse(image_recon, ep_frames) * weighting
    return loss

def collate(batch):
    agent_pos = torch.zeros((BATCH_SIZE, NR_EPS, MAX_LENGTH), dtype=torch.long)
    frames = torch.zeros((BATCH_SIZE, NR_EPS, MAX_LENGTH, IMG_CHANNELS, IMG_SZ, IMG_SZ), dtype=torch.float)
    padding = torch.ones((BATCH_SIZE, NR_EPS, MAX_LENGTH)).bool()
    for item_idx, item in enumerate(batch):
        for ep in range(NR_EPS):
            item_l = item[0][ep].shape[0] - 1
            agent_pos[item_idx, ep, :item_l] = item[0][ep][1:]
            frames[item_idx, ep, :item_l + 1] = item[1][ep]
            padding[item_idx, ep, item_l:] = False
    return agent_pos, padding, frames