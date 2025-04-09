import torch, PIL, os
import matplotlib.pyplot as plt
import models, utils
import pandas as pd
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

utils.seed_everything(utils.CFG["SEED"])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CAM_root = "C:\\Users\\PC00\\Desktop\\GradCAM\\"
os.makedirs(CAM_root, exist_ok=True)

input("To start GradCAM analysis, please press Enter: ")

"""     TWO cases    """
# (1) NORMAL vs SJS
# (2) NONSJS vs SJS
control_group = input("Select CONTROL group: ")
control_root = os.path.join(CAM_root, f"{control_group}_SJS")

utils.seed_everything(utils.CFG["SEED"])

PG_ResNet = models.ResNet().to(device)
PG_ResNet.eval()
SG_ResNet = models.ResNet().to(device)
SG_ResNet.eval()

PG_target_layer = PG_ResNet.model.layer4[2].conv3
SG_target_layer = SG_ResNet.model.layer4[2].conv3

for gland in os.listdir(control_root):
    if os.path.isdir(os.path.join(control_root, gland)):
        gland_path = os.path.join(control_root, gland)
        for img_name in os.listdir(gland_path):
            curr_ID = img_name.split("_")[0]
            curr_gland = img_name.split("_")[1].split(".")[0]
            image = read_image(os.path.join(gland_path, img_name))
            CAM_name = os.path.join(gland_path, f"{curr_ID}_{curr_gland}_GC.jpg")
            input_tensor = normalize(resize(image, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
            if curr_gland == "PTG":
                with SmoothGradCAMpp(PG_ResNet, PG_target_layer) as cam_extractor:
                    out = PG_ResNet(input_tensor.unsqueeze(0))
                    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            else:
                with SmoothGradCAMpp(SG_ResNet, SG_target_layer) as cam_extractor:
                    out = SG_ResNet(input_tensor.unsqueeze(0))
                    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            result = overlay_mask(PIL.Image.open(os.path.join(gland_path, img_name)), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.figure(); plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.savefig(CAM_name); plt.close()