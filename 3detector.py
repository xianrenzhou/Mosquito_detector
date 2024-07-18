'''
这是一个三分类器程序探索
用于识别background,audio.mosquqito
训练和验证的数据样本数量：
mosquito: 15850
background: 9635
audio: 479

测试样本数量：
mosquito: 1500
background: 1500
audio: 200
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split,Subset
import torch.nn.functional as F
import multiprocessing
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import imageio
# 网络结构定义
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据加载函数
def load_data(batch_size=32, split_ratio=0.7):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    num_workers = multiprocessing.cpu_count()
    # 加载整个数据集
    full_dataset = datasets.ImageFolder(root='3class_narrow', transform=transform)
    # targets = full_dataset.targets
    # train_indices = []
    # for class_index in range(len(full_dataset.classes)):
    #     class_mask = (np.array(targets) == class_index)
    #     class_indices = np.where(class_mask)[0]
    #     class_train_indices = np.random.choice(class_indices, 350, replace=False)
    #     train_indices.extend(class_train_indices)
    # # 从剩余样本中随机选取120个样本作为验证集
    # remaining_indices = list(set(range(len(full_dataset))) - set(train_indices))
    # val_indices = np.random.choice(remaining_indices, 120, replace=False)
    # train_dataset = Subset(full_dataset, train_indices)
    # val_dataset = Subset(full_dataset, val_indices)
  
    test_dataset = datasets.ImageFolder(root='test_narrow', transform=transform)
    # 计算训练集和测试集的大小
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # 随机切分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 加载数据集到 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# 训练模型函数
import matplotlib.pyplot as plt
import torch

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import imageio
import torch


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import imageio
import os
import torch

# 平滑数据的函数
def smooth_data(data):
    return gaussian_filter1d(data, sigma=1.0)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1000, device=torch.device("cpu"), save_path="model_1.pth"):
    model.train()  # 设置模型为训练模式
    plt.rcParams.update({'font.size': 25})
    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0

    # 初始化用于绘图的列表
    train_losses = []
    val_losses = []

    # 设置实时更新的图表
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    filenames = []  # 用于保存生成的图片文件名列表
    
    for epoch in range(num_epochs):
        train_loss, train_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)

        scheduler.step()  # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: current learning rate is {current_lr}")

        train_losses.append(train_loss / train_total)

        # 在验证集上评估
        val_loss, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_total += labels.size(0)

        val_losses.append(val_loss / val_total)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/train_total:.4f}, Val Loss: {val_loss/val_total:.4f}')

        # 更新图表
        ax.cla()
        ax.plot(smooth_data(train_losses), label='Train Loss')
        ax.plot(smooth_data(val_losses), label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

        plt.pause(0.1)
        
        # 保存当前图像
        filename = f'plot_epoch_{epoch+1}.png'
        filenames.append(filename)
        plt.savefig(filename)

        # 早停机制的检查和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0  # reset the trigger
            torch.save(model.state_dict(), save_path)
            print("Model saved as validation loss decreased.")
        else:
            trigger_times += 1
            print(f"Validation loss did not decrease, trigger times: {trigger_times}")

        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break  # early stop if val loss doesn't decrease for 'patience' times

    plt.ioff()

    # 生成GIF
    with imageio.get_writer('training_validation_loss.gif', mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # 删除生成的图片文件
    for filename in filenames:
        os.remove(filename)








# 测试模型函数
def test_model(model, test_loader, device=torch.device("cpu")):
    model.eval()  # 设置模型为评估模式
    total = correct = 0
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)



# 主函数
def main():

    # for i in range(50):
    #     print(f'第{i}个模型')
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     train_loader, val_loader, test_loader = load_data()
    #     model = CNNModel().to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch学习率乘以0.1
    #     train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, device=device,num_epochs=25,save_path="./model/model_"+str(i)+".pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_data()
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率乘以0.1
    train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, device=device,num_epochs=25,save_path="./model/model_2.pth")    
    test_model(model, test_loader, device=device)


if __name__ == '__main__':
    main()
