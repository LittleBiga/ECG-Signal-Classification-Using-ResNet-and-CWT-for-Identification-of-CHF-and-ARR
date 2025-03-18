# 标准库导入
import site
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             recall_score, precision_score,
                             classification_report)

# 第三方库导入
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

# 打印安装路径
print(site.getsitepackages())

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径
file_path = "ecgdataset"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = datasets.ImageFolder(root=file_path, transform=transform)
labels = [label for _, label in dataset.samples]

# 数据集划分
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    range(len(dataset.samples)),
    labels,
    test_size=(2 + 2) / 10,
    random_state=0,
    stratify=labels
)

val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices,
    temp_labels,
    test_size=2 / (2 + 2),
    random_state=0,
    stratify=temp_labels
)

# 创建数据集子集
train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)
test_subset = Subset(dataset, test_indices)

print('Number of training images: ', len(train_subset))
print('Number of validation images: ', len(val_subset))
print('Number of testing images: ', len(test_subset))

# 数据加载器配置
BATCH_SIZE = 20
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet30(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet30, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        # Layer 2
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        # Layer 3
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        # Layer 4
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 实例化模型
model = ResNet30().to(device)
num_class = 3

# 损失函数和优化器
class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 绘制损失曲线和准确率曲线
def plot_curves(train_losses, val_accuracies_per_class, f_measure_vals, g_mean_vals):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_class):
        if i == 0:
            label_name = 'ARR'
        elif i == 1:
            label_name = 'CHF'
        elif i == 2:
            label_name = 'NSR'
        plt.plot(val_accuracies_per_class[i], label=f'Validation Accuracy ({label_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curves per Class')
    plt.legend()
    plt.show()

    # 绘制F-measure曲线和G-mean曲线
    plt.figure(figsize=(12, 6))
    for i in range(num_class):
        if i == 0:
            label_name = 'ARR'
        elif i == 1:
            label_name = 'CHF'
        elif i == 2:
            label_name = 'NSR'
        plt.plot(f_measure_vals[i], label=f'F-measure ({label_name})')
    plt.plot(g_mean_vals, label='G-mean', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('F-measure and G-mean Curves')
    plt.legend()
    plt.show()


# 训练模型
# 训练参数
num_epochs = 30
train_losses = []
val_accuracies_per_class = [[] for _ in range(num_class)]
f_measure_vals = [[] for _ in range(num_class)]
g_mean_vals = []

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 记录训练损失
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 验证阶段
    model.eval()
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    # 计算验证集精确率、召回率和混淆矩阵
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    # val_accuracies.append(accuracy_val)

    # 计算每个类别的验证准确率
    for i in range(num_class):
        class_mask = (torch.tensor(y_true_val) == i)
        class_pred = torch.tensor(y_pred_val)[class_mask]
        class_true = torch.tensor(y_true_val)[class_mask]
        accuracy = accuracy_score(class_true, class_pred)
        val_accuracies_per_class[i].append(accuracy)

    # 计算每个类别的精确率和召回率
    precision_val = precision_score(y_true_val, y_pred_val, average=None)
    recall_val = recall_score(y_true_val, y_pred_val, average=None)

    # 计算特异度
    confusion_val = confusion_matrix(y_true_val, y_pred_val)
    specificity_val = []
    for i in range(len(confusion_val)):
        true_negative = sum(confusion_val[j][j] for j in range(len(confusion_val)) if j != i)
        false_positive = sum(confusion_val[j][i] for j in range(len(confusion_val)) if j != i)
        false_negative = sum(confusion_val[i][j] for j in range(len(confusion_val)) if j != i)
        specificity_val.append(true_negative / (true_negative + false_positive + false_negative))

    # 计算每个类别的 F-measure
    f_measure_val = f1_score(y_true_val, y_pred_val, average=None)
    for i in range(num_class):
        f_measure_vals[i].append(f_measure_val[i])

    # 计算 G-mean
    recall_per_class = recall_val
    specificity_per_class = []
    for i in range(len(confusion_val)):
        true_negative = sum(confusion_val[j][j] for j in range(len(confusion_val)) if j != i)
        false_positive = sum(confusion_val[j][i] for j in range(len(confusion_val)) if j != i)
        false_negative = sum(confusion_val[i][j] for j in range(len(confusion_val)) if j != i)
        specificity_per_class.append(true_negative / (true_negative + false_positive + false_negative))
    g_mean_val = (recall_per_class[0] * specificity_per_class[0] * recall_per_class[1] * specificity_per_class[1] *
                  recall_per_class[2] * specificity_per_class[2]) ** (1 / 6)
    g_mean_vals.append(g_mean_val)

    # 输出结果
    print(f'Validation Accuracy: {accuracy_val}')
    print(f'Validation Precision: {precision_val}')
    print(f'Validation Recall: {recall_val}')
    print(f'Validation Specificity: {specificity_val}')
    print(f'Validation F-measure: {f_measure_val}')
    print(f'Validation G-mean: {g_mean_val}')

# 测试模型
model.eval()
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())
# 计算测试集精确率、召回率和混淆矩阵
accuracy_test = accuracy_score(y_true_test, y_pred_test)

# 计算每个类别的精确率和召回率
precision_test = precision_score(y_true_test, y_pred_test, average=None)
recall_test = recall_score(y_true_test, y_pred_test, average=None)

# 计算特异度
confusion_test = confusion_matrix(y_true_test, y_pred_test)
specificity_test = []
for i in range(len(confusion_test)):
    true_negative = sum(confusion_test[j][j] for j in range(len(confusion_test)) if j != i)
    false_positive = sum(confusion_test[j][i] for j in range(len(confusion_test)) if j != i)
    false_negative = sum(confusion_test[i][j] for j in range(len(confusion_test)) if j != i)
    specificity_test.append(true_negative / (true_negative + false_positive + false_negative))

# 计算每个类别的 F-measure
f_measure_test = f1_score(y_true_test, y_pred_test, average=None)

# 计算 G-mean
recall_per_class_test = recall_test
specificity_per_class_test = []
for i in range(len(confusion_test)):
    true_negative = sum(confusion_test[j][j] for j in range(len(confusion_test)) if j != i)
    false_positive = sum(confusion_test[j][i] for j in range(len(confusion_test)) if j != i)
    false_negative = sum(confusion_test[i][j] for j in range(len(confusion_test)) if j != i)
    specificity_per_class_test.append(true_negative / (true_negative + false_positive + false_negative))
g_mean_test = (recall_per_class_test[0] * specificity_per_class_test[0] * recall_per_class_test[1] *
               specificity_per_class_test[1] * recall_per_class_test[2] * specificity_per_class_test[2]) ** (1 / 6)

print('\n')
print(f'Test Accuracy: {accuracy_test}')
print(f'Test Precision: {precision_test}')
print(f'Test Recall: {recall_test}')
print(f'Test Specificity: {specificity_test}')
print(f'Test F-measure: {f_measure_test}')
print(f'Test G-mean: {g_mean_test}')
print(f'Test Confusion Matrix:')
print(confusion_test)

# 绘制曲线
# plot_curves(train_losses, val_accuracies)
plot_curves(train_losses, val_accuracies_per_class, f_measure_vals, g_mean_vals)