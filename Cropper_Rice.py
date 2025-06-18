import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
from warnings import filterwarnings
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sn
from torchviz import make_dot
from tqdm import tqdm
from torch.nn import init

filterwarnings('ignore')



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = np.random.randint(1, 10000)

print(f"Random Seed: {seed}")

set_seed(seed)


# 数据加载和预处理

train_dir = "D:/FYPdataset/Unique2/RiceSmall/train80"
val_dir = "D:/FYPdataset/Unique2/RiceSmall/validation20"
test_dir = "D:/FYPdataset/Unique2/RiceSmall/Test"
labels = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy',
              'leaf_blast', 'leaf_scald', 'narrow_brown_spot']


class RiceLeafDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        for cls_name in class_names:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

    def get_filepath(self, idx):
        return self.image_paths[idx]



train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=0.2, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = RiceLeafDataset(train_dir, transform=train_transform)
val_dataset = RiceLeafDataset(val_dir, transform=val_transform)
test_dataset = RiceLeafDataset(test_dir, transform=val_transform)
batch_size = 4


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    return images, labels, paths

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=custom_collate)


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
        return out + residual



class EnsembleModel(nn.Module):
    def __init__(self, num_classes=6):
        super(EnsembleModel, self).__init__()
        self.model1 = models.inception_v3(pretrained=True)
        self.model1.fc = nn.Identity()
        self.model2 = models.efficientnet_v2_s(pretrained=True)
        self.model2.classifier = nn.Identity()
        self.model3 = models.resnet34(pretrained=True)
        self.model3.fc = nn.Identity()


        self.fc1 = nn.Linear(2048, 512)  # InceptionV3
        self.fc2 = nn.Linear(1280, 512)  # EfficientNetV2
        self.fc3 = nn.Linear(512, 512)   # ResNet34


        self.attn1 = CBAMBlock(512,reduction=16)
        self.attn2 = CBAMBlock(512,reduction=16)
        self.attn3 = CBAMBlock(512,reduction=16)


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
        out1 = self.model1(x)[0] if self.training else self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        out1 = self.fc1(out1.view(out1.size(0), -1))
        out2 = self.fc2(out2.view(out2.size(0), -1))
        out3 = self.fc3(out3.view(out3.size(0), -1))

        out1 = self.attn1(out1.unsqueeze(2).unsqueeze(3))
        out2 = self.attn2(out2.unsqueeze(2).unsqueeze(3))
        out3 = self.attn3(out3.unsqueeze(2).unsqueeze(3))

        out = out1+out2+out3

        return self.classifier(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnsembleModel().to(device)
criterion = nn.CrossEntropyLoss()

fc_layers = [model.fc1, model.fc2, model.fc3]
classifier_linears = []
for layer in model.classifier:
    if isinstance(layer, nn.Linear):
        classifier_linears.append(layer)
fc_layers.extend(classifier_linears)

fc_params = []
for layer in fc_layers:
    fc_params += list(layer.parameters())

other_params = [param for param in model.parameters() if param not in set(fc_params)]

param_groups = [
    {'params': fc_params, 'weight_decay': 1e-4},
    {'params': other_params, 'weight_decay': 0}
]
optimizer = optim.Adam(param_groups, lr=1e-5)



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



def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, batch_labels,_ in loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
    return 100. * correct / total


def calculate_metrics(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []
    for i in range(len(labels)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = TN / (TN + FP + 1e-7)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        specificity_list.append(specificity)

    return {
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'specificity': np.mean(specificity_list)
    }


history = {
    'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
    'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_specificity': [],
    'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_specificity': []
}


def evaluate_with_metrics(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_true = []
    all_probs = []
    all_files = []

    with (torch.no_grad()):
        for inputs, batch_labels, paths in loader:
            inputs = inputs.to(device)
            batch_labels= batch_labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(batch_labels.cpu().numpy())
            all_files.extend(paths)

    metrics = calculate_metrics(np.array(all_true), np.array(all_preds))

    correct = (np.array(all_preds) == np.array(all_true)).sum()
    metrics['accuracy'] = 100.0 * correct / len(all_true)

    wrong_indices = np.where(np.array(all_preds) != np.array(all_true))[0]
    metrics['wrong_files'] = [(all_files[i],
                               labels[all_true[i]],
                               all_preds[i])
                              for i in wrong_indices]
    metrics['auroc'] = roc_auc_score(all_true,
                                   all_probs,
                                   multi_class='ovr',
                                   average='macro')
    return running_loss / len(loader), metrics


num_epochs = 100
best_val_loss = float('inf')
patience = 5
patience_counter = 0
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_metrics = evaluate_with_metrics(model, val_loader, criterion)
    val_acc = evaluate(model, val_loader)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Calculate others
    train_metrics = calculate_metrics(
        np.array([label for _, label, _ in train_dataset]),
        np.array([model(input.unsqueeze(0).to(device)).argmax().cpu() for input, _, _ in train_dataset])
    )
    history['train_precision'].append(train_metrics['precision'])
    history['train_recall'].append(train_metrics['recall'])
    history['train_f1'].append(train_metrics['f1'])
    history['train_specificity'].append(train_metrics['specificity'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_specificity'].append(val_metrics['specificity'])
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Val Metrics - Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}, '
          f'F1: {val_metrics["f1"]:.4f}, Specificity: {val_metrics["specificity"]:.4f}')
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_SmallRice_SUM_p5_loss.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break


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
    plt.savefig('model_precision_recall.png', dpi=1200)
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


# CM
def plot_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())

    cm = confusion_matrix(all_true, all_preds)
    class_labels = ['bacterial_leaf_blight', 'brown_spot', 'healthy',
                    'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

    df_cm = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    plt.figure(figsize=(10, 10))
    sn.set_theme(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, cmap='summer', fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('model_confusion_matrix.png', dpi=200)
    plt.show()


plot_metrics(history)
plot_confusion_matrix(model, test_loader)
# Test Evaluation
model.load_state_dict(torch.load('best_model_SmallRice_SUM_p5_loss.pth'))
test_loss, test_metrics = evaluate_with_metrics(model, test_loader, criterion)
test_acc = evaluate(model, test_loader)
print("\nFinal Test Results:")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test F1 Score: {test_metrics['f1']:.4f}")
print(f"Test Specificity: {test_metrics['specificity']:.4f}")
print(f"Test AUROC: {test_metrics['auroc']:.4f}")


