import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()
        
        self.model.zero_grad()
        loss = output[:, target_class]
        loss.backward()
        
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1)
        cam = np.maximum(cam, 0)
        cam = cam[0]
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)

def overlay_cam_on_image(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[2].conv3  # Last convolutional layer
    grad_cam = GradCAM(model, target_layer)
    
    img_path = "your_image.jpg"
    input_tensor = preprocess_image(img_path)
    cam = grad_cam.generate_cam(input_tensor)
    overlay_cam_on_image(img_path, cam)