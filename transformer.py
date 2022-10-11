import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import nn

from bib_vt.settings import MAX_LENGTH, IMG_SZ, IMG_CHANNELS, IMG_SZ_COMPRESSED, TOP_K, NR_GPUs

"""Basic Transformer components taken from vit.py file in the vit-pytorch repository 
by lucidrains (https://github.com/lucidrains/vit-pytorch)"""

# helpers

def create_pos_encoding(b_size, max_length, img_size=IMG_SZ_COMPRESSED + 1, block_size=1, stride=1):
    res = (torch.arange(0, img_size - stride, stride) + block_size / 2).unsqueeze(0) / img_size
    res_x = res.repeat(int(img_size / stride) - 1, 1)
    res_y = res_x.T
    res_x = res_x.unsqueeze(0).repeat(b_size, 1, 1).contiguous().view(b_size, -1, 1).repeat(max_length, 1, 1)
    res_y = res_y.unsqueeze(0).repeat(b_size, 1, 1).contiguous().view(b_size, -1, 1).repeat(max_length, 1, 1)
    return res_x, res_y


# classes

class Encoder(nn.Module):

    # Constructor
    def __init__(self, out_channels=30):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=2 * (out_channels + 2), kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=2 * (out_channels + 2), out_channels=out_channels, kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Prediction
    def forward(self, x):
        x = functional.relu(self.pool(self.conv1(x)))
        x = functional.relu(self.pool(self.conv2(x)))
        return x


class Decoder(nn.Module):

    # Constructor
    def __init__(self, in_channels=32):
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(in_channels, in_channels * 2, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels * 2, IMG_CHANNELS, kernel_size=2, stride=2)

    # Prediction
    def forward(self, x):
        x = functional.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, hist, **kwargs):
        return self.fn(self.norm(x), self.norm(hist), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_head),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        if context is not None:
            qkv_context = self.to_qkv(context).chunk(3, dim=-1)
            _, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_context)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, device, dropout=0.1):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for l_idx, (attn, ff) in enumerate(self.layers):
            att_x = attn(x)
            x = att_x + x
            x = ff(x) + x
        return x


class CrossAttTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, device, dropout=0.1):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([])
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                CrossPreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim_head, FeedForward(dim_head, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, other):
        for l_idx, (attn, ff) in enumerate(self.layers):
            att_x = attn(x, other)
            x = att_x + x
            x = ff(x) + x
        return x


class VideoTransformer(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0., device, batch_size):
        super().__init__()
        self.dim = dim
        self.encoder = Encoder(out_channels=self.dim - 2)
        self.decoder = Decoder(in_channels=self.dim)
        self.batch_size = batch_size // NR_GPUs
        self.device = device
        self.empty_frame = torch.zeros(self.batch_size, 1, IMG_SZ_COMPRESSED ** 2, self.dim)
        self.self_att_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, self.device, dropout)
        self.past_steps_transformer = CrossAttTransformer(dim_head, depth, heads, dim_head, mlp_dim, self.device,
                                                          dropout)
        self.past_eps_transformer = CrossAttTransformer(dim_head, depth, heads, dim_head, mlp_dim, self.device, dropout)
        self.x_pos, self.y_pos = create_pos_encoding(b_size=self.batch_size, max_length=MAX_LENGTH)
        self.out = nn.Linear(dim_head, 1)

    def process_past_eps(self, past_ep_frames, batch_size, x_pos, y_pos, empty_frame, max_length, padding, curr_ep_enc):
        x_agg = torch.zeros(past_ep_frames.shape[1], batch_size, IMG_SZ_COMPRESSED ** 2, self.dim).to(self.device)
        for ep in range(past_ep_frames.shape[1]):
            past_ep = past_ep_frames[:, ep].reshape(-1, IMG_CHANNELS, IMG_SZ, IMG_SZ)
            past_ep_enc, past_ep_diff = self.encode_frames(past_ep, x_pos, y_pos, batch_size, max_length, empty_frame)
            for b in range(batch_size):
                past_ep_diff_b = past_ep_diff[b, 1:][padding[b, ep, :-1]]
                largest_diffs = torch.topk(past_ep_diff_b, TOP_K, dim=1)[1]
                idxes = torch.arange(past_ep_diff_b.shape[0]).unsqueeze(-1)
                trace = (past_ep_enc[b, 1:][padding[b, ep, :-1]][idxes, largest_diffs]).reshape(-1, self.dim).unsqueeze(
                    0)
                x_agg[ep, b] = self.past_eps_transformer(curr_ep_enc[b, :1], trace)
        return x_agg

    def to_device(self, curr_ep, padding, past_eps):
        curr_ep = curr_ep.to(self.device).reshape(-1, IMG_CHANNELS, IMG_SZ, IMG_SZ)
        padding = padding.to(self.device)
        x_pos = self.x_pos.to(self.device)
        y_pos = self.y_pos.to(self.device)
        empty_frame = self.empty_frame.to(self.device)
        past_eps = past_eps.to(self.device)
        return curr_ep, padding, x_pos, y_pos, empty_frame, past_eps

    def process_curr_ep(self, batch_size, max_length, diff_ims, padding, encoded, x):
        x_agg = torch.zeros(batch_size, max_length, IMG_SZ_COMPRESSED ** 2, self.dim).to(self.device)
        for b in range(batch_size):
            curr_ep_diff = diff_ims[b, 1:][padding[b, -1, 1:]]
            largest_diffs = torch.topk(curr_ep_diff, 3, dim=1)[1]
            idxes = torch.arange(curr_ep_diff.shape[0]).unsqueeze(-1)
            traces = (encoded[b, 1:][padding[b, -1, 1:]][idxes, largest_diffs]).unsqueeze(0).repeat(max_length, 1, 1, 1)
            mask = torch.tril(torch.ones(traces.shape[0], traces.shape[1]), diagonal=-1)
            mask = mask.unsqueeze(-1).unsqueeze(-1).to(self.device)
            traces = (traces * mask).reshape(max_length, -1, self.dim)
            x_agg[b] = self.past_steps_transformer(x[b], traces)
        return x_agg

    def encode_frames(self, frames, x_pos, y_pos, batch_size, max_length, empty_frame):
        encoded = self.encoder(frames).permute(0, 2, 3, 1).reshape(-1, IMG_SZ_COMPRESSED ** 2, self.dim - 2)
        encoded = torch.cat([encoded, x_pos, y_pos], dim=2)
        encoded = encoded.reshape(batch_size, max_length, IMG_SZ_COMPRESSED ** 2, self.dim)
        diff_ims = abs(encoded - torch.cat([empty_frame, encoded[:, :-1]], dim=1)).sum(-1)
        diff_ims = diff_ims - torch.median(diff_ims, dim=-1, keepdim=True)[0]
        return encoded, diff_ims

    def forward(self, curr_ep, padding, past_eps, max_length=MAX_LENGTH):
        batch_size = curr_ep.shape[0]
        curr_ep, padding, x_pos, y_pos, empty_frame, past_eps = self.to_device(curr_ep, padding, past_eps)
        encoded, diff_ims = self.encode_frames(curr_ep, x_pos, y_pos, batch_size, max_length, empty_frame)
        x_agg = self.process_past_eps(past_eps, batch_size, x_pos, y_pos, empty_frame, max_length, padding, encoded)
        x = x_agg.mean(0)
        x = self.self_att_transformer(x)
        x = x.unsqueeze(1).repeat(1, max_length, 1, 1)
        x_agg = self.process_curr_ep(batch_size, max_length, diff_ims, padding, encoded, x)
        recon = self.decoder(x_agg.reshape(-1, IMG_SZ_COMPRESSED, IMG_SZ_COMPRESSED, self.dim).permute(0, 3, 1, 2))
        pred = self.out(x_agg)
        return pred, recon
