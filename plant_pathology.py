

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  

class CFG:
    train_csv   = f"input/train.csv"
    test_csv    = f"input/test.csv"
    sample_sub  = f"output/sample_submission.csv"
    image_dir   = f"input/images"

    # Model
    model_name = "resnet50"   
    img_size    = 380                  
    num_classes = 4                   

    # Training
    epochs      = 15
    batch_size  = 16                   
    lr          = 1e-4
    weight_decay= 1e-5
    n_folds     = 5
    seed        = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlantDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test   = is_test

        # Label columns
        self.label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"] + ".jpg")
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image

        # Labels are one-hot: e.g. [1, 0, 0, 0]
        labels = torch.tensor(row[self.label_cols].values.astype(float),
                              dtype=torch.float32)
        return image, labels



train_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # ImageNet normalization — required for pretrained EfficientNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class PlantModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        # Load pretrained EfficientNet backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)              # raw logits
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()

            # Sigmoid converts logits → probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Mean column-wise AUC (exactly what Kaggle scores)
    aucs = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        aucs.append(auc)
    mean_auc = np.mean(aucs)

    return total_loss / len(loader), mean_auc


def main():
    train_df = pd.read_csv(CFG.train_csv)
    test_df  = pd.read_csv(CFG.test_csv)
    sub_df   = pd.read_csv(CFG.sample_sub)

    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(f"Label distribution:\n{train_df[['healthy','multiple_diseases','rust','scab']].sum()}\n")
    print(f"Using device: {CFG.device}")

    label_cols    = ["healthy", "multiple_diseases", "rust", "scab"]
    train_df["fold_label"] = train_df[label_cols].values.argmax(axis=1)

    oof_preds   = np.zeros((len(train_df), CFG.num_classes))
    test_preds  = np.zeros((len(test_df),  CFG.num_classes))

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True,
                          random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["fold_label"])):

        print(f"\n{'='*40}")
        print(f"  FOLD {fold + 1} / {CFG.n_folds}")
        print(f"{'='*40}")

        fold_train = train_df.iloc[train_idx]
        fold_val   = train_df.iloc[val_idx]

        train_ds = PlantDataset(fold_train, CFG.image_dir, train_transform)
        val_ds   = PlantDataset(fold_val,   CFG.image_dir, val_transform)
        test_ds  = PlantDataset(test_df,    CFG.image_dir, val_transform,
                                is_test=True)

        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size,
                                  shuffle=True,  num_workers=2,
                                  pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size,
                                  shuffle=False, num_workers=2,
                                  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size,
                                  shuffle=False, num_workers=2,
                                  pin_memory=True)

        model     = PlantModel(CFG.model_name, CFG.num_classes).to(CFG.device)
        # BCEWithLogitsLoss = Sigmoid + Binary Cross Entropy
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr,
                               weight_decay=CFG.weight_decay)
        # Cosine annealing: smoothly reduces LR → helps convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.epochs, eta_min=1e-6)

        best_auc   = 0.0
        best_path  = f"best_model_fold{fold}.pth"

        for epoch in range(1, CFG.epochs + 1):
            train_loss          = train_one_epoch(model, train_loader,
                                                  optimizer, criterion,
                                                  CFG.device)
            val_loss, val_auc   = validate(model, val_loader, criterion,
                                           CFG.device)
            scheduler.step()

            print(f"  Epoch {epoch:02d}/{CFG.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")

            # Save best model for this fold
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")

        print(f"\n  Loading best fold model (AUC: {best_auc:.4f}) for inference...")
        model.load_state_dict(torch.load(best_path))
        model.eval()

        # OOF predictions (for local AUC calculation)
        fold_oof = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(CFG.device)
                preds  = torch.sigmoid(model(images)).cpu().numpy()
                fold_oof.append(preds)
        oof_preds[val_idx] = np.concatenate(fold_oof, axis=0)

        # Test predictions (average across folds)
        fold_test = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(CFG.device)
                preds  = torch.sigmoid(model(images)).cpu().numpy()
                fold_test.append(preds)
        test_preds += np.concatenate(fold_test, axis=0) / CFG.n_folds

    oof_aucs = []
    for i, col in enumerate(label_cols):
        auc = roc_auc_score(train_df[col].values, oof_preds[:, i])
        oof_aucs.append(auc)
        print(f"OOF AUC [{col}]: {auc:.4f}")

    print(f"\n Overall OOF Mean AUC: {np.mean(oof_aucs):.4f}")

    sub_df[label_cols] = test_preds
    sub_df.to_csv("submission.csv", index=False)
    print(sub_df.head())


if __name__ == "__main__":
    main()
