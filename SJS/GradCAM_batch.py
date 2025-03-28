import torch, PIL, os
import matplotlib.pyplot as plt
import models, utils
import torchvision.transforms as transforms
import pandas as pd
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask

utils.seed_everything(utils.CFG["SEED"])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CAM_root = "E:\\Results\\SJS\\Figures\\CAM\\Batch"
os.makedirs(CAM_root, exist_ok=True)

input("To start GradCAM analysis, please press Enter: ")

"""     TWO cases    """
# (1) NORMAL vs SJS
# (2) NONSJS vs SJS
control_group = input("Select CONTROL group: ")

data_path = f".\\Correct_IDs({control_group}).xlsx"

df = pd.read_excel(data_path)
total_data = df["Path"]
total_class = df["Class"]

utils.seed_everything(utils.CFG["SEED"])

resize = transforms.Compose([
    transforms.Resize((utils.CFG['IMG_SIZE'], utils.CFG['IMG_SIZE'])),
    transforms.ToTensor()
])

PG_ResNet = models.ResNet().to(device)
PG_ResNet.load_state_dict(torch.load(utils.CFG["pg_res_path"], map_location="cuda"))
PG_ResNet.eval()
SG_ResNet = models.ResNet().to(device)
SG_ResNet.load_state_dict(torch.load(utils.CFG["sg_res_path"], map_location="cuda"))
SG_ResNet.eval()

total_preds, total_labels = [], []
for i, datapath in enumerate(total_data):
    filename = os.path.basename(datapath)
    curr_ID = filename.split("_")[1].split("-")[0]
    curr_gland = filename.split("_")[0]
    curr_class = total_class[i]
    save_folder = os.path.join(CAM_root, f"{control_group}_SJS", curr_class)
    os.makedirs(save_folder, exist_ok=True)
    image = resize(PIL.Image.open(datapath)).to(torch.float32).unsqueeze(0).to(device)
    if curr_gland == "PTG":
        image_name = os.path.join(save_folder, f"{curr_ID}_{curr_gland}.jpg")
        with SmoothGradCAMpp(PG_ResNet) as cam_extractor:
            out = PG_ResNet(image)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        image_name = os.path.join(save_folder, f"{curr_ID}_{curr_gland}.jpg")
        with SmoothGradCAMpp(SG_ResNet) as cam_extractor:
            out = SG_ResNet(image)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    result = overlay_mask(PIL.Image.open(datapath), to_pil_image(activation_map[0].squeeze(0), mode="F"), alpha=0.5)
    plt.figure(); plt.imshow(result); plt.axis("off"); plt.tight_layout(); plt.savefig(image_name); plt.close()