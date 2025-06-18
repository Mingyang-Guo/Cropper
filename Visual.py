import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from numpy import resize
from torchvision import transforms, models
import os
import torch.nn.functional as F
from torch.nn import init



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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)

        channel_weights = self.ca(x)


        weighted_feature = self.sa(channel_weights)


        return weighted_feature.view(weighted_feature.size(0), weighted_feature.size(1))



class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EnsembleModel, self).__init__()
        self.model1 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model1.fc = nn.Identity()
        self.model2 = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model2.classifier = nn.Identity()
        self.model3 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.model3.fc = nn.Identity()

        self.fc1 = nn.Linear(2048, 512)  # InceptionV3
        self.fc2 = nn.Linear(1280, 512)  # EfficientNetV2
        self.fc3 = nn.Linear(512, 512)  # ResNet34

        self.attn1 = CBAMBlock(512, reduction=16)
        self.attn2 = CBAMBlock(512, reduction=16)
        self.attn3 = CBAMBlock(512, reduction=16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512 * 3, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, num_classes)
        )

        self.classifier2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, num_classes)
        )

        self.backbone_classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out1 = self.model1(x)[0] if self.training else self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        out1 = self.fc1(out1.view(out1.size(0), -1))
        out2 = self.fc2(out2.view(out2.size(0), -1))
        out3 = self.fc3(out3.view(out3.size(0), -1))

        pred1 = out1
        pred2 = out2
        pred3 = out3


        combined = torch.cat([out1, out2, out3], dim=1)
        combined = combined.view(1, 512 * 3, 1, 1)

        out = torch.cat([out1, out2, out3], dim=1)

        final_output = self.classifier(out)
        pred1 = self.classifier2(pred1)
        pred2 = self.classifier2(pred2)
        pred3 = self.classifier2(pred3)
        return final_output, pred1, pred2, pred3, combined


class AttentionVisualizer(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        self.model1 = original_model.model1
        self.model2 = original_model.model2
        self.model3 = original_model.model3


        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2
        self.fc3 = original_model.fc3


        self.fc11 = nn.Linear(2048 * 8 * 8, 512)
        self.fc22 = nn.Linear(1280 * 10 * 10, 512)
        self.fc33 = nn.Linear(512 * 10 * 10, 512)


        self.attn_modules = {
            'InceptionV3': original_model.attn1,
            'EfficientNetV2': original_model.attn2,
            'ResNet34': original_model.attn3
        }

    def get_final_heatmap(self, x):

        with torch.no_grad():

            out1 = self.model1(x.clone())[0]
            out2 = self.model2(x.clone())
            out3 = self.model3(x.clone())

            out1 = out1.unsqueeze(0) if out1.dim() == 1 else out1


            out1 = self.fc1(out1.view(out1.size(0), -1))
            out2 = self.fc2(out2.view(out2.size(0), -1))
            out3 = self.fc3(out3.view(out3.size(0), -1))

            combined = torch.cat([out1, out2, out3], dim=1)




            if combined.dim() == 2:
                combined = combined.unsqueeze(2).unsqueeze(3)
                print("Reshaped combined:", combined.shape)


            heatmap = torch.mean(combined, dim=1, keepdim=True)



            if heatmap.size(2) == 1 and heatmap.size(3) == 1:
                heatmap = torch.nn.functional.interpolate(heatmap, size=(299, 299), mode='bilinear',
                                                          align_corners=False)
                print("Resized heatmap:", heatmap.shape)


            heatmap = heatmap.view(1, 1, 299, 299)

            return heatmap.cpu().numpy()

    def get_attention_weights(self, x):

        with torch.no_grad():

            x1 = x.clone()

            for name, module in self.model1.named_children():
                if name == 'Mixed_7c':
                    x1 = module(x1)
                    break
                if name not in ['fc', 'AuxLogits']:
                    x1 = module(x1)


            x2 = x.clone()

            x2 = self.model2.features(x2)


            x3 = x.clone()

            for name, module in self.model3.named_children():
                if name == 'avgpool':
                    break
                x3 = module(x3)


        x1_flat = x1.view(x1.size(0), -1)

        logits1 = self.fc11(x1_flat)




        x2_flat = x2.view(x2.size(0), -1)

        logits2 = self.fc22(x2_flat)


        x3_flat = x3.view(x3.size(0), -1)

        logits3 = self.fc33(x3_flat)


        attention_maps = {}


        spatial_map1 = torch.mean(x1, dim=1).detach().cpu().numpy()[0]
        spatial_map1 = (spatial_map1 - spatial_map1.min()) / (spatial_map1.max() - spatial_map1.min() + 1e-8)
        spatial_map_tensor = torch.from_numpy(spatial_map1).unsqueeze(0).unsqueeze(0)
        spatial_map1_resized = F.interpolate(spatial_map_tensor,
                                             size=(299, 299),
                                             mode='bilinear',
                                             align_corners=False).squeeze().numpy()


        spatial_map2 = torch.mean(x2, dim=1).detach().cpu().numpy()[0]
        spatial_map2 = (spatial_map2 - spatial_map2.min()) / (spatial_map2.max() - spatial_map2.min() + 1e-8)
        spatial_map_tensor = torch.from_numpy(spatial_map2).unsqueeze(0).unsqueeze(0)
        spatial_map2_resized = F.interpolate(spatial_map_tensor,
                                             size=(299, 299),
                                             mode='bilinear',
                                             align_corners=False).squeeze().numpy()


        spatial_map3 = torch.mean(x3, dim=1).detach().cpu().numpy()[0]
        spatial_map3 = (spatial_map3 - spatial_map3.min()) / (spatial_map3.max() - spatial_map3.min() + 1e-8)
        spatial_map_tensor = torch.from_numpy(spatial_map3).unsqueeze(0).unsqueeze(0)
        spatial_map3_resized = F.interpolate(spatial_map_tensor,
                                             size=(299, 299),
                                             mode='bilinear',
                                             align_corners=False).squeeze().numpy()


        prob1 = F.softmax(logits1, dim=1).cpu().numpy()[0]
        prob2 = F.softmax(logits2, dim=1).cpu().numpy()[0]
        prob3 = F.softmax(logits3, dim=1).cpu().numpy()[0]


        attention_maps['InceptionV3'] = {
            'channel': torch.ones(512).numpy(),
            'spatial': spatial_map1_resized,
            'predictions': prob1
        }

        attention_maps['EfficientNetV2'] = {
            'channel': torch.ones(512).numpy(),
            'spatial': spatial_map2_resized,
            'predictions': prob2
        }

        attention_maps['ResNet34'] = {
            'channel': torch.ones(512).numpy(),
            'spatial': spatial_map3_resized,
            'predictions': prob3
        }

        return attention_maps



transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def denormalize(tensor):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return tensor * std + mean


def visualize_ensemble_attention(image_path, model, device):

    raw_img = Image.open(image_path).convert('RGB')
    tensor = transform(raw_img).unsqueeze(0).to(device)


    with torch.no_grad():
        attention_maps = model.get_attention_weights(tensor)
        final_pred, pred1, pred2, pred3, comblined = original_model(tensor)


    fig = plt.figure(figsize=(18, 12))
    columns = 4
    rows = 3


    ax = fig.add_subplot(rows, columns, 1)
    ax.imshow(raw_img)
    ax.set_title('Original Image', fontproperties='SimHei')
    ax.axis('off')



    for idx, (model_name, maps) in enumerate(attention_maps.items(), start=1):

        ax = fig.add_subplot(rows, columns, idx * columns - 2)
        ax.imshow(maps['spatial'], cmap='jet')
        ax.set_title(f'{model_name} Attention')
        ax.axis('off')


        ax = fig.add_subplot(rows, columns, idx * columns - 1)
        img_array = denormalize(tensor.cpu().squeeze()).permute(1, 2, 0).numpy()
        ax.imshow(img_array)
        ax.imshow(maps['spatial'], cmap='jet', alpha=0.5)
        ax.set_title(f'{model_name} Overlay')
        ax.axis('off')


    spatial1 = attention_maps['InceptionV3']['spatial']
    spatial2 = attention_maps['EfficientNetV2']['spatial']
    spatial3 = attention_maps['ResNet34']['spatial']


    prob1 = attention_maps['InceptionV3']['predictions'][1]
    prob2 = attention_maps['EfficientNetV2']['predictions'][1]
    prob3 = attention_maps['ResNet34']['predictions'][1]
    total_prob = prob1 + prob2 + prob3 + 1e-8
    w1, w2, w3 = prob1 / total_prob, prob2 / total_prob, prob3 / total_prob

    final_heatmap = w1 * spatial1 + w2 * spatial2 + w3 * spatial3
    final_heatmap = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min() + 1e-8)


    ax = fig.add_subplot(rows, columns, columns * rows)
    img_array = denormalize(tensor.cpu().squeeze()).permute(1, 2, 0).numpy()
    ax.imshow(img_array)
    ax.imshow(final_heatmap, cmap='jet')
    ax.set_title('Final Integrated Heatmap')
    ax.axis('off')


    ax = fig.add_subplot(rows, 1, rows)
    ax.axis('off')
    final_probs = F.softmax(final_pred, dim=1).squeeze().cpu().numpy()
    text = "Final Ensemble Predictions:\n"
    for i, prob in enumerate(final_probs):
        text += f"Class {i}: {prob * 100:.1f}%\n"
    ax.text(0.875, 1.25, text, ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"viz_{os.path.basename(image_path)}")
    plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_CAT_loss.pth"   # The model path.   The concat method is used. Not sum.


    original_model = EnsembleModel().to(device)
    original_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    original_model.eval()


    visualizer = AttentionVisualizer(original_model).to(device)
    visualizer.eval()


    image_paths = [
        #### input the path of image ####
    ]


    for path in image_paths:
        if os.path.exists(path):
            print(f"Visualizing: {os.path.basename(path)}")
            visualize_ensemble_attention(path, visualizer, device)
        else:
            print(f"File not found: {path}")