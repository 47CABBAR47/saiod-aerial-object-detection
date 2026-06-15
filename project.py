import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# ==========================================
# 1. PARAMETRELER VE CİHAZ YAPILANDIRMASI
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
batch_size = 16
epochs = 100
learning_rate = 0.00001
early_stopping_patience = 10
early_stopping_min_delta = 0.0
split_seed = 42
freeze_hybrid_feature_extractors = False

output_dir = "egitim_sonuclari"
os.makedirs(output_dir, exist_ok=True)


# ==========================================
# 2. VERİ ÖN İŞLEME VE BÖLÜNME (%80, %10, %10)
# ==========================================
# Hocanın istediği gibi tüm veriler tek bir ana klasörde toplanmalı
base_data_path = r'C:\Users\omerf\OneDrive\Desktop\Deneme3\SAIOD\TrgVal'
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Veri setini tek bir havuz olarak indeksliyoruz
base_dataset = datasets.ImageFolder(base_data_path)
total_size = len(base_dataset)

# Oranların hesaplanması (Madde 2)
train_size = int(0.80 * total_size)
val_size = int(0.10 * total_size)
test_size = total_size - train_size - val_size

# Fiziksel ayırma yerine kod içinde bölünme
split_generator = torch.Generator().manual_seed(split_seed)
indices = torch.randperm(total_size, generator=split_generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset = Subset(datasets.ImageFolder(base_data_path, transform=train_transform), train_indices)
val_dataset = Subset(datasets.ImageFolder(base_data_path, transform=eval_transform), val_indices)
test_dataset = Subset(datasets.ImageFolder(base_data_path, transform=eval_transform), test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = base_dataset.classes
num_classes = len(class_names)

CNN_MODELS = {
    'resnet18',
    'resnet50',
    'efficientnet_b0',
    'mobilenet_v3_small',
    'densenet121',
    'convnext_tiny',
}

TRANSFORMER_MODELS = {
    'vit_b_16',
    'vit_b_32',
    'swin_t',
    'swin_v2_t',
}


# ==========================================
# 3. MODELLERİN OLUŞTURULMASI
# ==========================================
def build_model(model_name):
    print(f"[MODEL] {model_name} mimarisi inşa ediliyor...")
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'swin_t':
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Bilinmeyen model adi: {model_name}")
    return model.to(device)


def get_model_family(model_name):
    if model_name in CNN_MODELS:
        return 'cnn'
    if model_name in TRANSFORMER_MODELS:
        return 'transformer'
    return 'hybrid' if model_name.startswith('hybrid_') else 'unknown'


def safe_dir_name(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def build_feature_extractor(model_name, weights_path=None):
    model = build_model(model_name)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))

    if model_name in {'resnet18', 'resnet50'}:
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == 'efficientnet_b0':
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif model_name == 'mobilenet_v3_small':
        feature_dim = model.classifier[3].in_features
        model.classifier = nn.Identity()
    elif model_name == 'densenet121':
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif model_name == 'convnext_tiny':
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
    elif model_name in {'vit_b_16', 'vit_b_32'}:
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif model_name in {'swin_t', 'swin_v2_t'}:
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    else:
        raise ValueError(f"Hibrit için bilinmeyen model tipi: {model_name}")

    return model, feature_dim


class HybridModel(nn.Module):
    def __init__(self, cnn_name, transformer_name, cnn_weights_path=None, transformer_weights_path=None, freeze_features=False):
        super().__init__()
        self.cnn_name = cnn_name
        self.transformer_name = transformer_name
        self.cnn, cnn_feature_dim = build_feature_extractor(cnn_name, cnn_weights_path)
        self.transformer, transformer_feature_dim = build_feature_extractor(transformer_name, transformer_weights_path)

        if freeze_features:
            for parameter in self.cnn.parameters():
                parameter.requires_grad = False
            for parameter in self.transformer.parameters():
                parameter.requires_grad = False

        combined_feature_dim = cnn_feature_dim + transformer_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        combined_features = torch.cat((cnn_features, transformer_features), dim=1)
        return self.classifier(combined_features)



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        patience: Kaç epoch boyunca iyileşme olmazsa durdurulsun?
        min_delta: İyileşme sayılması için gereken minimum değişim.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True
        # Eğer yeni loss, eskiden (best - delta) kadar küçük değilse (yani iyileşme yoksa)
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            # İyileşme var! En iyi değeri güncelle ve sayacı sıfırla.
            self.best_loss = val_loss
            self.counter = 0
            return True
# ==========================================
# 4. EĞİTİM, CANLI TAKİP VE RAPORLAMA FONKSİYONU
# ==========================================
def calculate_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    if len(labels) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }

    return {
        'accuracy': float((preds == labels).mean()),
        'precision': float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall': float(recall_score(labels, preds, average='macro', zero_division=0)),
        'f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
    }


def compact_metrics(metrics):
    return {
        'loss': float(metrics['loss']),
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
    }


def train_one_epoch(model, loader, criterion, optimizer, epoch, model_name):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds, all_labels = [], []

    progress = tqdm(loader, desc=f"{model_name} Epoch {epoch}/{epochs} [Eğitim]", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size_now = images.size(0)
        running_loss += loss.item() * batch_size_now
        total += batch_size_now

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.2%}")

    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / max(total, 1)
    metrics['preds'] = np.array(all_preds)
    metrics['labels'] = np.array(all_labels)
    return metrics


def evaluate_model(model, loader, criterion, phase_name, show_progress=True):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds, all_labels = [], []

    iterator = tqdm(loader, desc=phase_name, leave=False) if show_progress else loader
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size_now = images.size(0)
            running_loss += loss.item() * batch_size_now
            total += batch_size_now

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            if show_progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.2%}")

    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / max(total, 1)
    metrics['preds'] = np.array(all_preds)
    metrics['labels'] = np.array(all_labels)
    return metrics


def metric_line(title, metrics):
    return (
        f"{title}: Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.2%} | "
        f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )


def write_epoch_report(path, epoch, train_metrics, val_metrics, test_metrics, early_stopping, validation_improved, best_checkpoint):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(metric_line("Train", train_metrics) + "\n")
        f.write(metric_line("Validation", val_metrics) + "\n")
        f.write(metric_line("Test", test_metrics) + "\n")
        f.write(f"EarlyStopping Counter: {early_stopping.counter}/{early_stopping.patience}\n")
        f.write(f"Best Val Loss: {early_stopping.best_loss:.4f}\n")
        f.write(f"Validation Loss Improved: {validation_improved}\n")
        f.write(f"Best Checkpoint By Val Accuracy: {best_checkpoint}\n")


def save_training_plot(history, model_name, model_res_dir):
    epochs_axis = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, history['train_loss'], label='Train Loss')
    plt.plot(epochs_axis, history['val_loss'], label='Val Loss')
    plt.plot(epochs_axis, history['test_loss'], label='Test Loss')
    plt.title(f'{model_name} Loss Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, history['train_acc'], label='Train Acc')
    plt.plot(epochs_axis, history['val_acc'], label='Val Acc')
    plt.plot(epochs_axis, history['test_acc'], label='Test Acc')
    plt.title(f'{model_name} Accuracy Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_res_dir, f"{model_name}_grafik.png"), dpi=150)
    plt.close()


def save_final_test_report(model_name, model_res_dir, test_metrics):
    report = classification_report(
        test_metrics['labels'],
        test_metrics['preds'],
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    with open(os.path.join(model_res_dir, "final_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(metric_line("Final Test", test_metrics) + "\n\n")
        f.write(report)

    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'], labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Final Test Confusion Matrix')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig(os.path.join(model_res_dir, "final_confusion_matrix.png"), dpi=150)
    plt.close()


def save_summary(results):
    rows = []
    for result in results:
        rows.append({
            'model': result['name'],
            'family': result['family'],
            'best_epoch': result['best_epoch'],
            'best_val_loss': result['best_val']['loss'],
            'best_val_acc': result['best_val']['accuracy'],
            'test_loss_at_best_epoch': result['best_test']['loss'],
            'test_acc_at_best_epoch': result['best_test']['accuracy'],
            'final_test_loss': result['final_test']['loss'],
            'final_test_acc': result['final_test']['accuracy'],
            'elapsed_seconds': result['elapsed_seconds'],
            'best_model_path': result['best_model_path'],
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "model_ozeti.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def train_single_model(name, model=None, family=None):
    start_time = time.time()
    family = family or get_model_family(name)
    model = build_model(name) if model is None else model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )

    safe_name = safe_dir_name(name)
    model_res_dir = os.path.join(output_dir, safe_name)
    os.makedirs(model_res_dir, exist_ok=True)
    best_model_path = os.path.join(model_res_dir, "best_model.pth")

    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
    }
    best_epoch = 0
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_snapshot = None

    print(f"\n" + "="*70)
    print(f"[BASLADI] {name.upper()} MODELİ EĞİTİMİ BAŞLADI | Aile: {family}")
    print("="*70)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch, name)
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            phase_name=f"{name} Epoch {epoch}/{epochs} [Doğrulama]",
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            criterion,
            phase_name=f"{name} Epoch {epoch}/{epochs} [Test]",
        )

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])

        validation_improved = early_stopping(val_metrics['loss'])
        best_checkpoint = (
            val_metrics['accuracy'] > best_val_acc
            or (
                np.isclose(val_metrics['accuracy'], best_val_acc)
                and val_metrics['loss'] < best_val_loss
            )
        )

        if best_checkpoint:
            best_epoch = epoch
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_snapshot = {
                'train': compact_metrics(train_metrics),
                'val': compact_metrics(val_metrics),
                'test': compact_metrics(test_metrics),
            }
            torch.save(model.state_dict(), best_model_path)

        print(f"[EPOCH] Epoch [{epoch:03d}/{epochs}] Sonu Bilgileri:")
        print(f"   {metric_line('Train', train_metrics)}")
        print(f"   {metric_line('Validation', val_metrics)}")
        print(f"   {metric_line('Test', test_metrics)}")
        print(f"   En iyi checkpoint : Epoch {best_epoch} | Val Acc: {best_val_acc:.2%} | Val Loss: {best_val_loss:.4f}")
        print(f"   Erken Durdurma    : Sayaç {early_stopping.counter}/{early_stopping.patience} | En iyi Val Loss: {early_stopping.best_loss:.4f}")
        print("   Durum             : Val loss iyileşti." if validation_improved else "   Durum             : Val loss iyileşmedi.")
        print("-" * 70)

        txt_path = os.path.join(model_res_dir, f"epoch_{epoch}.txt")
        write_epoch_report(
            txt_path,
            epoch,
            train_metrics,
            val_metrics,
            test_metrics,
            early_stopping,
            validation_improved,
            best_checkpoint,
        )

        if early_stopping.early_stop:
            print(f"[DURDU] Gelişim durduğu için {epoch}. epochta eğitim kesildi!")
            break

    save_training_plot(history, safe_name, model_res_dir)

    print(f"[FINAL TEST] {name} için en iyi checkpoint ile final test yapılıyor...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        phase_name=f"{name} [Final Test]",
    )
    save_final_test_report(safe_name, model_res_dir, final_test_metrics)

    if best_snapshot is None:
        best_snapshot = {
            'train': compact_metrics(train_metrics),
            'val': compact_metrics(val_metrics),
            'test': compact_metrics(test_metrics),
        }

    elapsed_seconds = time.time() - start_time
    result = {
        'name': name,
        'family': family,
        'model_dir': model_res_dir,
        'best_model_path': best_model_path,
        'best_epoch': best_epoch,
        'best_train': best_snapshot['train'],
        'best_val': best_snapshot['val'],
        'best_test': best_snapshot['test'],
        'final_test': compact_metrics(final_test_metrics),
        'elapsed_seconds': elapsed_seconds,
    }

    print(f"[TAMAM] {name} tamamlandı.")
    print(f"   Seçim metriği     : Best Val Acc {result['best_val']['accuracy']:.2%} (Epoch {best_epoch})")
    print(f"   Final Test Acc    : {result['final_test']['accuracy']:.2%}")
    print(f"   Kayıt klasörü     : {model_res_dir}")
    return result


def select_best_model(results, family):
    candidates = [result for result in results if result['family'] == family]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda result: (
            result['best_val']['accuracy'],
            -result['best_val']['loss'],
        )
    )


def write_hybrid_selection(best_cnn, best_transformer):
    selection_path = os.path.join(output_dir, "hibrit_secim_ozeti.txt")
    with open(selection_path, "w", encoding="utf-8") as f:
        f.write("Hibrit model seçimi validation accuracy değerine göre yapılmıştır.\n")
        f.write("Test metrikleri raporlanır; model seçimi için kullanılmaz.\n\n")
        f.write(f"En iyi CNN: {best_cnn['name']} | Epoch: {best_cnn['best_epoch']} | Val Acc: {best_cnn['best_val']['accuracy']:.4f} | Final Test Acc: {best_cnn['final_test']['accuracy']:.4f}\n")
        f.write(f"En iyi Transformer: {best_transformer['name']} | Epoch: {best_transformer['best_epoch']} | Val Acc: {best_transformer['best_val']['accuracy']:.4f} | Final Test Acc: {best_transformer['final_test']['accuracy']:.4f}\n")
    return selection_path


def train_and_analyze(model_list):
    results = []
    for name in model_list:
        result = train_single_model(name, family=get_model_family(name))
        results.append(result)
        save_summary(results)

    best_cnn = select_best_model(results, 'cnn')
    best_transformer = select_best_model(results, 'transformer')

    if best_cnn is None or best_transformer is None:
        print("[UYARI] Hibrit model atlandı: Hem CNN hem Transformer ailesinden en az bir model gerekli.")
        return results

    selection_path = write_hybrid_selection(best_cnn, best_transformer)
    print("\n" + "="*70)
    print("[SECIM] HİBRİT MODEL İÇİN SEÇİLENLER")
    print(f"   CNN         : {best_cnn['name']} | Best Val Acc: {best_cnn['best_val']['accuracy']:.2%}")
    print(f"   Transformer : {best_transformer['name']} | Best Val Acc: {best_transformer['best_val']['accuracy']:.2%}")
    print("   Not         : Seçim validation accuracy ile yapıldı; test sonucu sadece raporlandı.")
    print(f"   Özet        : {selection_path}")
    print("="*70)

    hybrid_name = f"hybrid_{best_cnn['name']}_{best_transformer['name']}"
    hybrid_model = HybridModel(
        cnn_name=best_cnn['name'],
        transformer_name=best_transformer['name'],
        cnn_weights_path=best_cnn['best_model_path'],
        transformer_weights_path=best_transformer['best_model_path'],
        freeze_features=freeze_hybrid_feature_extractors,
    )
    hybrid_result = train_single_model(hybrid_name, model=hybrid_model, family='hybrid')
    results.append(hybrid_result)
    save_summary(results)
    return results

if __name__ == '__main__':
    models_to_test = [
        # CNN modelleri
        'resnet18',
        'resnet50',
        'efficientnet_b0',
        'mobilenet_v3_small',
        'densenet121',
        'convnext_tiny',
        # Transformer modelleri
        'vit_b_16',
        'vit_b_32',
        'swin_t',
        'swin_v2_t',
    ] # Test etmek istediğin modeller
    train_and_analyze(models_to_test)
