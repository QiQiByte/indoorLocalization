import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import get_dataloaders
from model import CSILocalizationModel
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

def main():
    config = {
        'num_classes': 317,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.01,
        'val_ratio': 0.2,
        # 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    }

    def load_train_coords(csv_path):
        df = pd.read_csv(csv_path)
        coords = df[['x', 'y']].values
        labels = df['class_id'].values - 1
        return coords, labels


    # 设置工作目录为项目根目录
    project_root = '/Users/baiqi/GitClone/PyCharm/indoor_localization'
    os.chdir(project_root)
    print(f"Current working directory: {os.getcwd()}")  # 调试用

    # 加载数据
    csv_path = os.path.join('dataset', 'lab', 'phase', 'images_labels.csv')
    train_coords, train_labels = load_train_coords(csv_path)

    # 数据加载器
    train_loader, val_loader = get_dataloaders(
        csv_path=csv_path,
        image_dir=os.path.join('dataset', 'lab', 'phase', 'images_test'),
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        # num_workers=4 if config['device'].type == 'cuda' else 0,
        num_workers=1 if config['device'].type == 'mps' else 0
    )

    model = CSILocalizationModel(
        num_classes=config['num_classes'],
        train_coords=train_coords,
        train_labels=train_labels
    ).to(config['device'])

    optimizer = optim.SGD(model.parameters(),
                          lr=config['learning_rate'],
                          momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_loc_err': [], 'val_loc_err': [],
        'lr': [],
        'log_var_cls': [], 'log_var_reg': []
    }

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        epoch_stats = {
            'train': {'loss': 0, 'correct': 0, 'total': 0, 'distances': []},
            'val': {'loss': 0, 'correct': 0, 'total': 0, 'distances': []}
        }

        for images, labels, coords in train_loader:
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            coords = coords.to(config['device'])

            optimizer.zero_grad()
            class_out, coord_out = model(images)

            batch_distances = model.calculate_distance(coord_out, coords)
            epoch_stats['train']['distances'].extend(batch_distances.cpu().numpy()*100)

            losses = model.get_loss(class_out, coord_out, labels, coords)
            losses['total_loss'].backward()
            optimizer.step()

            epoch_stats['train']['loss'] += losses['total_loss'].item()
            epoch_stats['train']['correct'] += (class_out.argmax(1) == labels).sum().item()
            epoch_stats['train']['total'] += labels.size(0)

        history['log_var_cls'].append(losses['log_var_cls'])
        history['log_var_reg'].append(losses['log_var_reg'])

        model.eval()
        with torch.no_grad():
            for images, labels, coords in val_loader:
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                coords = coords.to(config['device'])

                class_out, coord_out = model(images)
                losses = model.get_loss(class_out, coord_out, labels, coords)

                batch_distances = model.calculate_distance(coord_out, coords)
                epoch_stats['val']['distances'].extend(batch_distances.cpu().numpy()*100)

                epoch_stats['val']['loss'] += losses['total_loss'].item()
                epoch_stats['val']['correct'] += (class_out.argmax(1) == labels).sum().item()
                epoch_stats['val']['total'] += labels.size(0)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        history['train_loss'].append(epoch_stats['train']['loss'] / len(train_loader))
        history['train_acc'].append(100 * epoch_stats['train']['correct'] / epoch_stats['train']['total'])
        history['train_loc_err'].append(np.mean(epoch_stats['train']['distances']))

        history['val_loss'].append(epoch_stats['val']['loss'] / len(val_loader))
        history['val_acc'].append(100 * epoch_stats['val']['correct'] / epoch_stats['val']['total'])
        history['val_loc_err'].append(np.mean(epoch_stats['val']['distances']))

        scheduler.step(history['val_loss'][-1])

        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"[Train] Loss: {history['train_loss'][-1]:.4f} | "
              f"Acc: {history['train_acc'][-1]:.2f}% | "
              f"LocErr: {history['train_loc_err'][-1]:.2f}cm")
        print(f"[Val]  Loss: {history['val_loss'][-1]:.4f} | "
              f"Acc: {history['val_acc'][-1]:.2f}% | "
              f"LocErr: {history['val_loc_err'][-1]:.2f}cm")
        print(f"LR: {current_lr:.6f}")
        print(f"log_var_cls: {losses['log_var_cls']:.4f} | log_var_reg: {losses['log_var_reg']:.4f}")

        if history['val_acc'][-1] > best_val_acc:
            best_val_acc = history['val_acc'][-1]
            torch.save(model.state_dict(), 'best_model.pth')
            print("🔥 Saved new best model (val_acc: {:.2f}%)".format(best_val_acc))

    torch.save(model.state_dict(), 'final_model.pth')
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(history['train_loc_err'], label='Train')
    plt.plot(history['val_loc_err'], label='Validation')
    plt.title('Localization Error (cm)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # 可视化 log_var 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history['log_var_cls'], label='log_var_cls')
    plt.plot(history['log_var_reg'], label='log_var_reg')
    plt.title('Log Variance of Tasks')
    plt.xlabel('Step')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('log_var_curve.png')
    plt.show()

    val_distances = np.array(epoch_stats['val']['distances'])
    sorted_errors = np.sort(val_distances)
    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, cdf, label='Validation Localization Error CDF')
    plt.xlabel('Localization Error (cm)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Localization Error on Validation Set')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cdf_localization_error.png')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()