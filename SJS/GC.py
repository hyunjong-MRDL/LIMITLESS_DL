import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()

        target = output[:, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        alpha = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(alpha * activations, axis=1)

        cam = np.maximum(cam, 0)
        cam = cam[0]

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))

        return cam

model = models.resnet50(pretrained=True)
model.eval()

target_layer = model.layer4[2].conv3
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "D:/Datasets/SJS/Processed/SJS/PTG_46512-0000.jpg"
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

output = model(input_tensor)
class_idx = torch.argmax(output).item()
cam = gradcam.generate_cam(input_tensor, class_idx)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
original_image = np.array(image.resize((224, 224)))
overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

plt.imshow(overlay)
plt.axis("off")
plt.title("Overlay")

plt.show()