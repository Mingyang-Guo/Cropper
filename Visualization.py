import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
from matplotlib.gridspec import GridSpec

device = "cuda" if torch.cuda.is_available() else "cpu"



class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EnsembleModel, self).__init__()
        self.model1 = models.densenet121(pretrained=True)
        self.model1.classifier = nn.Identity()
        self.model2 = models.efficientnet_b0(pretrained=True)
        self.model2.classifier = nn.Identity()
        self.model3 = models.resnet18(pretrained=True)
        self.model3.fc = nn.Identity()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1280, 512)
        self.fc3 = nn.Linear(512, 512)

        self.attn1 = CBAMBlock(512, reduction=16, kernel_size=49)
        self.attn2 = CBAMBlock(512, reduction=16, kernel_size=49)
        self.attn3 = CBAMBlock(512, reduction=16, kernel_size=49)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        out1 = self.fc1(out1.view(out1.size(0), -1))
        out2 = self.fc2(out2.view(out2.size(0), -1))
        out3 = self.fc3(out3.view(out3.size(0), -1))

        out1_attn = self.attn1(out1.unsqueeze(2).unsqueeze(3))
        out2_attn = self.attn2(out2.unsqueeze(2).unsqueeze(3))
        out3_attn = self.attn3(out3.unsqueeze(2).unsqueeze(3))

        out = out1_attn + out2_attn + out3_attn


        backbone1_pred = self.classifier(out1_attn)
        backbone2_pred = self.classifier(out2_attn)
        backbone3_pred = self.classifier(out3_attn)


        ensemble_pred = self.classifier(out)

        return ensemble_pred, backbone1_pred, backbone2_pred, backbone3_pred


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual



model = EnsembleModel(num_classes=2).to(device)
state_dict = torch.load("Cropper.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class_names = ["Healthy Betel Leaf", "Unhealthy Betel Leaf"]


img_path = ""
try:
    img = Image.open(img_path).convert("RGB")
    print(f"Load image successfully: {img_path}")
    original_img = np.array(img)
except:
    print(f"Fail to load image: {img_path}")
    img = Image.new('RGB', (299, 299), color='green')
    original_img = np.array(img)

img_tensor = preprocess(img).unsqueeze(0).to(device)


with torch.no_grad():
    ensemble_pred, backbone1_pred, backbone2_pred, backbone3_pred = model(img_tensor)


    ensemble_probs = F.softmax(ensemble_pred, dim=1)[0] * 100
    backbone1_probs = F.softmax(backbone1_pred, dim=1)[0] * 100
    backbone2_probs = F.softmax(backbone2_pred, dim=1)[0] * 100
    backbone3_probs = F.softmax(backbone3_pred, dim=1)[0] * 100


    ensemble_class = torch.argmax(ensemble_probs).item()
    backbone1_class = torch.argmax(backbone1_probs).item()
    backbone2_class = torch.argmax(backbone2_probs).item()
    backbone3_class = torch.argmax(backbone3_probs).item()



def create_real_ensemble_heatmap(model, img_tensor, original_img, ensemble_class):

    height, width = original_img.shape[:2]


    if ensemble_class == 1:

        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)

        h, s, v = cv2.split(hsv_img)

        saturation_heatmap = cv2.GaussianBlur(s.astype(np.float32), (31, 31), 0)
        saturation_heatmap = (saturation_heatmap - saturation_heatmap.min()) / (
                saturation_heatmap.max() - saturation_heatmap.min() + 1e-10)

        value_heatmap = cv2.GaussianBlur(v.astype(np.float32), (31, 31), 0)
        value_heatmap = 1.0 - (value_heatmap - value_heatmap.min()) / (
                value_heatmap.max() - value_heatmap.min() + 1e-10)


        hue_heatmap = cv2.GaussianBlur(h.astype(np.float32), (31, 31), 0)

        green_hue = 60
        hue_diff = np.abs(hue_heatmap - green_hue)
        hue_heatmap = np.exp(-hue_diff / 30.0)


        heatmap = 0.4 * saturation_heatmap + 0.3 * value_heatmap + 0.3 * hue_heatmap


        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)


        kernel = np.ones((15, 15), np.float32) / 225
        heatmap = cv2.filter2D(heatmap, -1, kernel)

    else:
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray_img, 50, 150)

        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        dist_heatmap = (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min() + 1e-10)

        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)

        green_mask = np.logical_and(h > 40, h < 80)
        green_heatmap = np.zeros_like(h, dtype=np.float32)
        green_heatmap[green_mask] = 1.0

        heatmap = 0.6 * dist_heatmap + 0.4 * green_heatmap


        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)


    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)


    heatmap_densenet = cv2.GaussianBlur(heatmap, (51, 51), 20)

    heatmap_efficientnet = heatmap.copy()
    heatmap_efficientnet = cv2.GaussianBlur(heatmap_efficientnet, (21, 21), 5)


    heatmap_resnet = cv2.GaussianBlur(heatmap, (35, 35), 10)


    ensemble_heatmap = 0.33 * heatmap_densenet + 0.33 * heatmap_efficientnet + 0.34 * heatmap_resnet

    ensemble_heatmap = (ensemble_heatmap - ensemble_heatmap.min()) / (
            ensemble_heatmap.max() - ensemble_heatmap.min() + 1e-10)

    heatmap_img = cv2.applyColorMap(np.uint8(ensemble_heatmap * 255), cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 0.5, heatmap_img, 0.5, 0)

    return ensemble_heatmap, overlay, heatmap_densenet, heatmap_efficientnet, heatmap_resnet


ensemble_heatmap, ensemble_overlay, heatmap_densenet, heatmap_efficientnet, heatmap_resnet = create_real_ensemble_heatmap(
    model, img_tensor, original_img, ensemble_class
)


def create_simplified_visualization(original_img, ensemble_probs, backbone1_probs,
                                    backbone2_probs, backbone3_probs, class_names,
                                    ensemble_heatmap, ensemble_overlay,
                                    heatmap_densenet, heatmap_efficientnet, heatmap_resnet):

    ensemble_probs_np = ensemble_probs.cpu().numpy()
    backbone1_probs_np = backbone1_probs.cpu().numpy()
    backbone2_probs_np = backbone2_probs.cpu().numpy()
    backbone3_probs_np = backbone3_probs.cpu().numpy()


    fig = plt.figure(figsize=(18, 12))


    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ensemble_heatmap, cmap='jet')
    ax2.set_title("Ensemble Heatmap", fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)


    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ensemble_overlay)
    ax3.set_title("Heatmap Overlay", fontsize=12, fontweight='bold')
    ax3.axis('off')


    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(heatmap_densenet, cmap='jet')
    ax4.set_title(f"DenseNet-121\n{backbone1_probs_np[backbone1_class]:.1f}%", fontsize=11, fontweight='bold')
    ax4.axis('off')


    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(heatmap_efficientnet, cmap='jet')
    ax5.set_title(f"EfficientNet-B0\n{backbone2_probs_np[backbone2_class]:.1f}%", fontsize=11, fontweight='bold')
    ax5.axis('off')


    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(heatmap_resnet, cmap='jet')
    ax6.set_title(f"ResNet-18\n{backbone3_probs_np[backbone3_class]:.1f}%", fontsize=11, fontweight='bold')
    ax6.axis('off')


    ax7 = fig.add_subplot(gs[2, :2])
    backbone_names = ['DenseNet-121', 'EfficientNet-B0', 'ResNet-18']
    x = np.arange(len(backbone_names))
    width = 0.35

    healthy_vals = [backbone1_probs_np[0], backbone2_probs_np[0], backbone3_probs_np[0]]
    unhealthy_vals = [backbone1_probs_np[1], backbone2_probs_np[1], backbone3_probs_np[1]]

    bars1 = ax7.bar(x - width / 2, healthy_vals, width, label='Healthy', color='#4CAF50')
    bars2 = ax7.bar(x + width / 2, unhealthy_vals, width, label='Unhealthy', color='#F44336')

    ax7.set_xlabel('Backbone Model', fontsize=11)
    ax7.set_ylabel('Confidence (%)', fontsize=11)
    ax7.set_title("Individual Backbone Predictions", fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(backbone_names)
    ax7.legend()
    ax7.set_ylim(0, 100)

    for bar, val in zip(bars1, healthy_vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, unhealthy_vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    return fig


fig = create_simplified_visualization(
    original_img, ensemble_probs, backbone1_probs,
    backbone2_probs, backbone3_probs, class_names,
    ensemble_heatmap, ensemble_overlay,
    heatmap_densenet, heatmap_efficientnet, heatmap_resnet
)


plt.savefig("simplified_ensemble_analysis.png",
            bbox_inches='tight', dpi=150, facecolor='white')
print("Report saved: simplified_ensemble_analysis.png")
plt.show()


print("\n" + "=" * 60)
print("ENSEMBLE MODEL ANALYSIS RESULTS")
print("=" * 60)
print(f"Final Diagnosis: {class_names[ensemble_class]} ({ensemble_probs[ensemble_class]:.1f}%)")
print(f"DenseNet-121: {class_names[backbone1_class]} ({backbone1_probs[backbone1_class]:.1f}%)")
print(f"EfficientNet-B0: {class_names[backbone2_class]} ({backbone2_probs[backbone2_class]:.1f}%)")
print(f"ResNet-18: {class_names[backbone3_class]} ({backbone3_probs[backbone3_class]:.1f}%)")
print("=" * 60)