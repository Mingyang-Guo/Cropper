import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class AgricultureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(data_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def create_vit_model(num_classes, model_name='vit_tiny_patch16_224'):

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model



class MetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.softmax(outputs, dim=1)

        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().detach().numpy())

    def compute_all_metrics(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)


        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)


        cm = confusion_matrix(targets, predictions)
        specificity = self._calculate_specificity(cm)


        mcc = self._calculate_mcc(cm)

        auroc = self._calculate_auroc(targets, probabilities)

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Specificity': specificity,
            'MCC': mcc,
            'AUROC': auroc
        }

        return metrics, cm

    def _calculate_specificity(self, cm):

        specificities = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
        return np.mean(specificities)

    def _calculate_mcc(self, cm):

        n = cm.sum()
        total = 0
        for k in range(len(cm)):
            for l in range(len(cm)):
                for m in range(len(cm)):
                    total += cm[k, k] * cm[m, l] - cm[l, k] * cm[k, m]

        denominator = 1
        for k in range(len(cm)):
            row_sum = cm[k, :].sum()
            col_sum = cm[:, k].sum()
            denominator *= (row_sum * col_sum) ** 0.5

        mcc = total / denominator if denominator != 0 else 0
        return mcc

    def _calculate_auroc(self, targets, probabilities):

        if self.num_classes == 2:

            auroc = roc_auc_score(targets, probabilities[:, 1])
        else:

            auroc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
        return auroc

    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.verbose:
                print(f'Val loss: {val_loss:.4f}')
        elif val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'Loss improved: {val_loss:.4f}, reset patience')
        else:

            self.counter += 1
            if self.verbose:
                print(
                    f'Loss: {val_loss:.4f} (Best: {self.best_loss:.4f}), patience counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True



def train_model(model, train_loader, val_loader, num_epochs=80, learning_rate=0.001, patience=8):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, verbose=True)

    best_val_acc = 0.0
    best_model_state = None

    print("training...")
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })


        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)


                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val Acc': f'{100 * val_correct / val_total:.2f}%'
                })

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total


        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train loss: {avg_train_loss:.4f}, train acc: {train_acc:.2f}%, '
              f'Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.2f}%')


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': avg_val_loss
            }, 'best_vit_model.pth')
            print(f'  best model saved! Val acc: {best_val_acc:.2f}%, val loss: {avg_val_loss:.4f}')


        early_stopping(avg_val_loss, epoch)
        if early_stopping.early_stop:
            print(f'\nearly stopping! stop at epoch {epoch + 1} ')
            print(f'Best val loss {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch + 1}')
            break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model



def evaluate_model(model, test_loader, class_names):
    model.eval()
    metrics_calculator = MetricsCalculator(len(class_names))

    print("testing...")
    test_pbar = tqdm(test_loader, desc='testing')
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            metrics_calculator.update(outputs, labels)


    metrics, confusion_mat = metrics_calculator.compute_all_metrics()


    print("\n" + "=" * 60)
    print(" test result :")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


    print("\nCM:")
    print(confusion_mat)


    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics, confusion_mat


def main():
    # input your own dataset address
    train_dir = ""
    val_dir = ""
    test_dir = ""


    for path, name in [(train_dir, "train dataset"), (val_dir, "valid dataset"), (test_dir, "test dataset")]:
        if not os.path.exists(path):
            print(f"error: {name}path {path} not exist!")
            return


    print("loading data...")
    train_dataset = AgricultureDataset(train_dir, transform=train_transform)
    val_dataset = AgricultureDataset(val_dir, transform=val_transform)
    test_dataset = AgricultureDataset(test_dir, transform=test_transform)


    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



    num_classes = len(train_dataset.classes)

    model = create_vit_model(num_classes=num_classes, model_name='vit_tiny_patch16_224')


    trained_model = train_model(
        model, train_loader, val_loader, num_epochs=80, learning_rate=0.001, patience=8
    )


    metrics, confusion_mat = evaluate_model(trained_model, test_loader, train_dataset.classes)


    torch.save(trained_model.state_dict(), 'final_vit_model.pth')
    print("\nmodel saved as 'final_vit_model.pth'")



def predict_image(model, image_path, class_names, transform):

    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return class_names[predicted_class], confidence


if __name__ == "__main__":
    main()