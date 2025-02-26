import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from Swin_Transformer import SwinTransformer
from VIT import VisionTransformer, get_b16_config
import logging  # 添加日志模块

# data_dir = '/kaggle/input/rice-plant-leaf-disease-classification-v1i-folder' # 在kaggle上运行，使用此路径

# 数据路径（本地 Windows 系统）
data_dir = './data'

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test']),
}

# 创建 DataLoader
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0),
}

# 数据集大小和类别
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"类别: {class_names}")
print(f"数据集大小: {dataset_sizes}")

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型
def initialize_model(model_type="swin"):
    if model_type == "swin":
        model = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
    elif model_type == "vit":
        config = get_b16_config()
        model = VisionTransformer(config, img_size=224, num_classes=num_classes)
    else:
        raise ValueError("Model type must be 'swin' or 'vit'")
    return model.to(device)

# 设置日志
def setup_logging(model_name):
    log_file = f'./{model_name.lower()}_log.txt'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='w'
    )
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return log_file

# 训练函数
def train_model(model, criterion, optimizer, num_epochs=35, patience=10, model_name="Model"):
    log_file = setup_logging(model_name)
    logging.info(f"Starting training for {model_name}")
    logging.info(f"Categories: {class_names}")
    logging.info(f"Dataset sizes: {dataset_sizes}")

    best_acc = 0.0
    best_model_wts = model.state_dict()
    epochs_no_improve = 0

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            data_loader = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=False)
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                data_loader.set_postfix({
                    'loss': running_loss / (data_loader.n + 1),
                    'acc': running_corrects.double() / (data_loader.n + 1) / inputs.size(0)
                })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc.item())

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                    logging.info(f"New best validation accuracy: {best_acc:.4f}")
                else:
                    epochs_no_improve += 1
                    logging.info(f'Validation accuracy did not improve, epochs without improvement: {epochs_no_improve}/{patience}')
                    if epochs_no_improve >= patience:
                        logging.info(f'Early stopping triggered! Stopping at epoch {epoch+1}, best validation accuracy: {best_acc:.4f}')
                        model.load_state_dict(best_model_wts)
                        visualize_training(train_losses, valid_losses, train_accs, valid_accs, model_name)
                        return model, train_losses, valid_losses, train_accs, valid_accs, best_acc

    logging.info(f'Training completed without early stopping, best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    visualize_training(train_losses, valid_losses, train_accs, valid_accs, model_name)
    return model, train_losses, valid_losses, train_accs, valid_accs, best_acc

# 测试函数
def test_model(model, model_name="Model"):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        data_loader = tqdm(dataloaders['test'], desc='Testing', leave=False)
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            data_loader.set_postfix({
                'acc': running_corrects.double() / (data_loader.n + 1) / inputs.size(0)
            })
    test_acc = running_corrects.double() / dataset_sizes['test']
    logging.info(f'{model_name} Test Acc: {test_acc:.4f}')
    return test_acc

# 可视化函数
def visualize_training(train_losses, valid_losses, train_accs, valid_accs, model_name="Model"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Valid Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, valid_accs, 'r-', label='Valid Acc')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = f'./{model_name.lower()}_training_curves.png'
    plt.savefig(save_path)
    logging.info(f"{model_name} training curves saved to {save_path}")
    plt.close()

# 主函数
def main():
    # 训练 Swin Transformer
    print("训练 Swin Transformer...")
    model_swin = initialize_model("swin")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_swin.parameters(), lr=0.0001)
    model_swin, train_losses_swin, valid_losses_swin, train_accs_swin, valid_accs_swin, best_acc_swin = train_model(
        model_swin, criterion, optimizer, num_epochs=35, patience=10, model_name="SwinTransformer"
    )
    torch.save(model_swin.state_dict(), './swin_transformer_rice.pth')
    logging.info("Swin Transformer model saved to ./swin_transformer_rice.pth")
    test_acc_swin = test_model(model_swin, "Swin Transformer")

    # 训练 ViT
    print("\n训练 ViT...")
    model_vit = initialize_model("vit")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_vit.parameters(), lr=0.0001)
    model_vit, train_losses_vit, valid_losses_vit, train_accs_vit, valid_accs_vit, best_acc_vit = train_model(
        model_vit, criterion, optimizer, num_epochs=35, patience=10, model_name="ViT"
    )
    torch.save(model_vit.state_dict(), './vit_rice.pth')
    logging.info("ViT model saved to ./vit_rice.pth")
    test_acc_vit = test_model(model_vit, "ViT")

    # 性能对比
    logging.info("\n性能对比:")
    logging.info(f"Swin Transformer - Best Validation Acc: {best_acc_swin:.4f}, Test Acc: {test_acc_swin:.4f}")
    logging.info(f"ViT - Best Validation Acc: {best_acc_vit:.4f}, Test Acc: {test_acc_vit:.4f}")
    logging.info(f"当前目录内容: {os.listdir('./')}")

if __name__ == "__main__":
    main()