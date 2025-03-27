import torch, PIL
import matplotlib.pyplot as plt
import data, models, utils
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask

utils.seed_everything(utils.CFG["SEED"])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root = "D:/Datasets/SJS/"
processed_root = f"{root}Processed/"
CAM_root = "E:/Results/SJS/Figures/CAM/Trained/"

input("To start GradCAM analysis, please press Enter: ")

"""     TWO cases    """
# (1) NORMAL vs SJS
# (2) NONSJS vs SJS
control_group = input("Select CONTROL group: ")

total_SJS = data.split_by_patient(data.load_by_diagnosis(processed_root, "SJS"))
total_CTR = data.split_by_patient(data.load_by_diagnosis(processed_root, control_group))

train_SJS, test_SJS = data.train_test_split(total_SJS, utils.CFG["TEST_PORTION"])
train_CTR, test_CTR = data.train_test_split(total_CTR, utils.CFG["TEST_PORTION"])

PG_ResNet = models.ResNet().to(device)
PG_VGG = models.VGG16().to(device)
SG_ResNet = models.ResNet().to(device)
SG_VGG = models.VGG16().to(device)

X_pg, X_sg, Y_pg, Y_sg = data.gland_dataset(test_SJS, test_CTR)

PG_ResNet.load_state_dict(torch.load(utils.CFG["pg_res_path"], map_location="cuda"))
PG_VGG.load_state_dict(torch.load(utils.CFG["pg_vgg_path"], map_location="cuda"))
SG_ResNet.load_state_dict(torch.load(utils.CFG["sg_res_path"], map_location="cuda"))
SG_VGG.load_state_dict(torch.load(utils.CFG["sg_vgg_path"], map_location="cuda"))
PG_ResNet.eval()
PG_VGG.eval()
SG_ResNet.eval()
SG_VGG.eval()
pg_patients = list(Y_pg.keys())
sg_patients = list(Y_sg.keys())

for image in selected_images:
    diagnosis = image.split("/")[4]
    gland = image.split("/")[-1].split("_")[0]
    ID = image.split("/")[-1].split("_")[1].split("-")[0]
    selected_model = selected_models[ID]

    if gland == "PTG":
        if selected_model == "ResNet":
            model = PG_ResNet
            target_layer = PG_RES_target_layer
        elif selected_model == "VGG":
            model = PG_VGG
            target_layer = PG_VGG_target_layer
        else: continue
    elif gland == "SMG":
        if selected_model == "ResNet":
            model = SG_ResNet
            target_layer = SG_RES_target_layer
        elif selected_model == "VGG":
            model = SG_VGG
            target_layer = SG_VGG_target_layer
        else: continue
    
    save_folder = f"{CAM_root}{utils.control_group}_SJS/{diagnosis}/"
    utils.createFolder(save_folder)
    
    image_name = f"{save_folder}{ID}_{gland}_{selected_model}.jpg"

    """(1) Transform the image and put it into the model."""
    transformed = utils.preprocessing(PIL.Image.open(image)).float().unsqueeze(0).to(device)
    
    with SmoothGradCAMpp(model) as cam_extractor:
        out = model(transformed)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    
    result = overlay_mask(PIL.Image.open(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    plt.figure(); plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.savefig(image_name); plt.close()