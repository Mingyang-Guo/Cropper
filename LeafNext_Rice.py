import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from torchvision import transforms, datasets
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
# from pytorch_grad_cam import LayerCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
import pandas as pd
import numpy as np
import math
import random
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn



class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.ca = ChannelAttention(dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.ca(x)
        x = input + self.drop_path(x)
        return x


class LeafConvNeXt(nn.Module):
    def __init__(self, num_classes=6, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        x = self.classifier(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 2. Data loading with augmentation and class balancing
class BalancedDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, samples_per_class=500, balance=True):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples_per_class = samples_per_class
        self.balance = balance

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            image_paths = [
                os.path.join(class_path, fname)
                for fname in os.listdir(class_path)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if self.balance:
                if len(image_paths) >= samples_per_class:
                    selected_paths = random.sample(image_paths, samples_per_class)
                else:
                    selected_paths = image_paths * (samples_per_class // len(image_paths)) + \
                                    random.sample(image_paths, samples_per_class % len(image_paths))
            else:
                selected_paths = image_paths

            self.samples.extend([(p, idx, p) for p in selected_paths])  # Include file path

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, file_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, file_path

train_dir = "/home/guest/mingyang/RiceSmall/train80"
val_dir = "/home/guest/mingyang/RiceSmall/validation20"
test_dir = "/home/guest/mingyang/RiceSmall/Test"

class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']
labels = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = BalancedDataset(train_dir, transform=train_transform, samples_per_class=500)
val_dataset = BalancedDataset(val_dir, transform=val_test_transform, samples_per_class=100)
test_dataset = BalancedDataset(test_dir, transform=val_test_transform, samples_per_class=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        n_classes = outputs.size(1)
        one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1.0)
        smooth_targets = one_hot * (1 - self.epsilon) + self.epsilon / n_classes
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss

class CosineAnnealingWarmupScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            cos_epoch = epoch - self.warmup_epochs
            cos_total = self.total_epochs - self.warmup_epochs
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_total))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_convnext(num_classes=6):
    base_model = models.convnext_tiny(weights='IMAGENET1K_V1')
    for param in base_model.features.parameters():
        param.requires_grad = False
    base_model.classifier[2] = nn.Linear(base_model.classifier[2].in_features, num_classes)
    return base_model

model = load_pretrained_convnext(num_classes=6).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.05)
scheduler = CosineAnnealingWarmupScheduler(optimizer, total_epochs=100, warmup_epochs=20, base_lr=0.004, min_lr=1e-6)

num_epochs = 100
criterion = LabelSmoothingLoss(epsilon=0.1)

history = {
    'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
    'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_specificity': [],
    'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_specificity': [], 'train_auroc': [],
    'val_auroc': []
}

best_val_loss = float('inf')
best_model_path = "best_leafconvnext_Rice.pth"


def calculate_metrics(y_true, y_pred, y_probs):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []
    auroc_list = []

    for i in range(min(cm.shape[0], len(labels))):
        TP = cm[i, i] if i < cm.shape[0] else 0
        FP = cm[:, i].sum() - TP if i < cm.shape[0] else 0
        FN = cm[i, :].sum() - TP if i < cm.shape[0] else 0
        TN = cm.sum() - (TP + FP + FN)

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = TN / (TN + FP + 1e-7)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        specificity_list.append(specificity)

        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        try:
            auroc = roc_auc_score(y_true_binary, y_score)
        except ValueError:
            auroc = np.nan
        auroc_list.append(auroc)

    return {
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'specificity': np.mean(specificity_list),
        'auroc': np.nanmean(auroc_list)
    }

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def evaluate_with_metrics(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    batch_losses = []
    all_preds = []
    all_true = []
    all_probs = []
    all_files = []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if not torch.all((labels >= 0) & (labels < 6)):
                raise ValueError(f"Invalid labels detected: {labels}")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            batch_losses.append(loss.item())
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_files.extend(paths)
    avg_loss = running_loss / len(loader)
    print(f"Batch losses: mean={np.mean(batch_losses):.4f}, std={np.std(batch_losses):.4f}, "
          f"min={np.min(batch_losses):.4f}, max={np.max(batch_losses):.4f}")
    metrics = calculate_metrics(np.array(all_true), np.array(all_preds), np.array(all_probs))
    metrics['accuracy'] = 100.0 * (np.array(all_preds) == np.array(all_true)).sum() / len(all_true)
    wrong_indices = np.where(np.array(all_preds) != np.array(all_true))[0]
    metrics['wrong_files'] = [(all_files[i], class_names[all_true[i]], all_preds[i]) for i in wrong_indices]
    return avg_loss, metrics

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, batch_labels, _ in loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
    return 100. * correct / total

model.to(device)
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_metrics = evaluate_with_metrics(model, val_loader, criterion)
    val_acc = evaluate(model, val_loader)

    model.eval()
    train_preds = []
    train_true = []
    train_probs = []
    with torch.no_grad():
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            train_probs.extend(probs.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
    train_metrics = calculate_metrics(np.array(train_true), np.array(train_preds), np.array(train_probs))

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    history['train_precision'].append(train_metrics['precision'])
    history['train_recall'].append(train_metrics['recall'])
    history['train_f1'].append(train_metrics['f1'])
    history['train_specificity'].append(train_metrics['specificity'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_specificity'].append(val_metrics['specificity'])
    history['val_auroc'].append(val_metrics['auroc'])

    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_leafconvnext_Rice.pth')

def plot_metrics(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.ravel()
    metrics = ['precision', 'recall']
    for i, met in enumerate(metrics):
        ax[i].plot(history[f'train_{met}'])
        ax[i].plot(history[f'val_{met}'])
        ax[i].set_title(f'Model {met.capitalize()}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].grid(color='#e0e0eb')
        ax[i].legend(['train', 'val'])
    plt.savefig('model_precision_recall_auroc.png', dpi=1200)
    plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.ravel()
    metrics = ['acc', 'loss']
    for i, met in enumerate(metrics):
        ax[i].plot(history[f'train_{met}'])
        ax[i].plot(history[f'val_{met}'])
        ax[i].set_title(f'Model {met.capitalize()}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].grid(color='#e0e0eb')
        ax[i].legend(['train', 'val'])
    plt.savefig('model_accuracy_loss.png', dpi=1200)
    plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.ravel()
    metrics = ['f1', 'specificity']
    for i, met in enumerate(metrics):
        ax[i].plot(history[f'train_{met}'])
        ax[i].plot(history[f'val_{met}'])
        ax[i].set_title(f'Model {met.upper()}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].grid(color='#e0e0eb')
        ax[i].legend(['train', 'val'])
    plt.savefig('model_f1_specificity.png', dpi=1200)
    plt.show()


def plot_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_true, all_preds)
    class_labels = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    plt.figure(figsize=(10, 10))
    sn.set_theme(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, cmap='summer', fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('model_confusion_matrix.png', dpi=200)
    plt.show()

# 7. Evaluation: compute metrics on test set
plot_metrics(history)
plot_confusion_matrix(model, test_loader)


model.load_state_dict(torch.load('best_leafconvnext_Rice.pth'))
test_loss, test_metrics = evaluate_with_metrics(model, test_loader, criterion)
test_acc = evaluate(model, test_loader)
print("\nFinal Test Results:")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test F1 Score: {test_metrics['f1']:.4f}")
print(f"Test Specificity: {test_metrics['specificity']:.4f}")
print(f"Test AUROC: {test_metrics['auroc']:.4f}")


plot_confusion_matrix(model, test_loader)