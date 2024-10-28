import torch, random

device = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameters = {"SEED": 42,
                   "EPOCH": 10,
                   "LR": 1e-4,
                   "model_save_path": }

def seed_everything(SEED):
  random.seed(SEED)
  np.random.seed(SEED)
  os.environ["PYTHONHASHSEED"]=str(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.benchmark=True

seed_everything(hyperparameters["SEED"])
