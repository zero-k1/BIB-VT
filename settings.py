import torch

NR_EPS = 9
NR_GPUs = torch.cuda.device_count()
BATCH_SIZE = 6
NR_WORKERS = NR_GPUs * 2
NR_EPOCHS = 6
VAL_INTERVAL = int(4000 / BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM = 32
DEPTH = 5
DIM_HEAD = 32
HEADS = 8
MLP_DIM = 256

MAX_LENGTH = 90
SUBSAMPLE_FACTOR = 5
TOP_K = 3

IMG_SZ = 84
IMG_SZ_COMPRESSED = 21
IMG_SZ_RAW = 200
IMG_CHANNELS = 3

data_path_train = '../data/bib_train'
data_path_eval = '../data/bib_eval'
model_dir = "../saved_models/"
model_name = "model.pt"

