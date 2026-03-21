import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from load import load_data, restore_labels
from nets import Normal_FNN_ReLU, Basic_FNN_ReLU, Deep_FNN_ReLU

import os

# ***************************************************
# 学习率的影响
#
# 实验设置：
# 固定网络结构: 2个隐藏层
# 固定激活函数: ReLU
# 测试4种学习率：0.1 0.01 0.001 0.0001
# 
# ***************************************************

random_seed = 42 # 设置固定随机种子

rescale_or_not = True # 是否标准化数据集

save_dir = "./model_weights" # 模型权重保存目录
can_save_model = os.path.isdir(save_dir) # 检查目录是否存在
if not can_save_model:
    os.makedirs(save_dir, exist_ok=True)

def train_model(which_model, learning_rate):
    # 1.超参数设置与设备选择
    EPOCHS = 100          # 训练总轮数
    BATCH_SIZE = 32       # 批次大小
    LEARNING_RATE = learning_rate # 学习率
    ID = str(learning_rate).replace('.', '_')
    
    # 自动检测并使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}\n")

    # 2.加载数据
    train_loader, valid_loader, test_loader, mean, std = load_data(
        test_size=0.2, 
        random_state=random_seed, 
        valid_size=0.2, 
        batch_size=BATCH_SIZE, 
        rescale=rescale_or_not
    )
    
    # 3.初始化模型、损失函数与优化器
    if which_model == "Basic_FNN_ReLU":
        model = Basic_FNN_ReLU().to(device)
    elif which_model == "Normal_FNN_ReLU":
        model = Normal_FNN_ReLU().to(device)
    elif which_model == "Deep_FNN_ReLU":
        model = Deep_FNN_ReLU().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 用于记录最佳的验证集 Loss，以便保存最优模型
    best_valid_loss = float('inf')

    # 4.训练
    print("-" * 40)
    
    Train_losses = []
    Valid_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        #训练
        model.train()
        train_loss_sum = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_X.size(0)
            
        # 计算平均训练 Loss
        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        Train_losses.append(avg_train_loss)
        
        #验证
        model.eval()
        valid_loss_sum = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                valid_loss_sum += loss.item() * batch_X.size(0)
                
        # 计算平均验证 Loss
        avg_valid_loss = valid_loss_sum / len(valid_loader.dataset)
        Valid_losses.append(avg_valid_loss)
        
        # 保存表现最好的模型权重
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), "{}/best_diabetes_{}.pth".format(save_dir, ID))
        
        # 打印进度
        print(f"Epoch [{epoch:3d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

    if can_save_model:
        print("已将最佳模型保存为 'best_diabetes_{}.pth'".format(ID))
    else:
        print("因为目录 '{}' 原本不存在，已新建同名文件夹。已将最佳模型保存为 'best_diabetes_{}.pth'".format(save_dir, ID))
    print("-" * 40)
    
    return Train_losses, Valid_losses

    
def evaluate_model(which_model, learning_rate):
    BATCH_SIZE = 32       # 批次大小
    ID = str(learning_rate).replace('.', '_')
    
    # 自动检测并使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.加载数据
    train_loader, valid_loader, test_loader_true, mean, std = load_data(
        test_size=0.2, 
        random_state=random_seed, 
        valid_size=0.2, 
        batch_size=BATCH_SIZE, 
        rescale=False #不标准化测试集
    )
    
    train_loader, valid_loader, test_loader_rescale, mean, std = load_data(
        test_size=0.2, 
        random_state=random_seed, 
        valid_size=0.2, 
        batch_size=BATCH_SIZE, 
        rescale=True #标准化测试集
    )
    
    # 3.初始化模型、损失函数与优化器
    if which_model == "Basic_FNN_ReLU":
        model = Basic_FNN_ReLU().to(device)
    elif which_model == "Normal_FNN_ReLU":
        model = Normal_FNN_ReLU().to(device)
    elif which_model == "Deep_FNN_ReLU":
        model = Deep_FNN_ReLU().to(device)

    criterion = nn.MSELoss()
    
    # 加载最佳权重
    model.load_state_dict(torch.load("{}/best_diabetes_{}.pth".format(save_dir, ID), weights_only=True))
    model.eval()
    
    if rescale_or_not:
        test_loss_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader_rescale:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss_sum += loss.item() * batch_X.size(0)
        # 计算平均测试 Loss
        avg_test_loss = test_loss_sum / len(test_loader_rescale.dataset)
        print("-" * 40)
        print(f"learning rate = {ID} 在测试集上的标签标准化后的 MSE Loss: {avg_test_loss:.4f}")
    
    errors_sum = 0.0
    with torch.no_grad():
        for batch_X, batch_y_true in test_loader_true:
            batch_X = batch_X.to(device)
            normalized_preds = model(batch_X)
            normalized_preds = normalized_preds.cpu()
            if rescale_or_not:
                real_preds = restore_labels(normalized_preds, mean, std)
            else:
                real_preds = normalized_preds

            # 计算绝对方均误差
            errors = criterion(real_preds, batch_y_true.cpu())
            errors_sum += errors.item() * batch_X.size(0)
            
    # 计算平均绝对方均误差
    avg_test_error = errors_sum / len(test_loader_true.dataset)
    print(f"learning rate = {ID} 在原始测试集上的 MSE Loss: {avg_test_error:.2f}")
    print("-" * 40)


if __name__ == "__main__":
    if rescale_or_not:
        print("本次实验使用了标签标准化后的数据集。")
    else:
        print("本次实验使用了标签未标准化的数据集。")
    
    import matplotlib.pyplot as plt
    
    Train_losses_1, Valid_losses_1 = train_model('Normal_FNN_ReLU', 0.1)
    Train_losses_2, Valid_losses_2 = train_model('Normal_FNN_ReLU', 0.01)
    Train_losses_3, Valid_losses_3 = train_model('Normal_FNN_ReLU', 0.001)
    Train_losses_4, Valid_losses_4 = train_model('Normal_FNN_ReLU', 0.0001)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    axes[0].plot(Train_losses_1, label='learning_rate=0.1', color='orange')
    axes[0].plot(Train_losses_2, label='learning_rate=0.01', color='green')
    axes[0].plot(Train_losses_3, label='learning_rate=0.001', color='blue')
    axes[0].plot(Train_losses_4, label='learning_rate=0.0001', color='red')

    axes[0].set_title('Training Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(Valid_losses_1, label='learning_rate=0.1', color='orange')
    axes[1].plot(Valid_losses_2, label='learning_rate=0.01', color='green')
    axes[1].plot(Valid_losses_3, label='learning_rate=0.001', color='blue')
    axes[1].plot(Valid_losses_4, label='learning_rate=0.0001', color='red')

    axes[1].set_title('Validation Loss Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.show()
    
    evaluate_model('Normal_FNN_ReLU', 0.1)
    evaluate_model('Normal_FNN_ReLU', 0.01)
    evaluate_model('Normal_FNN_ReLU', 0.001)
    evaluate_model('Normal_FNN_ReLU', 0.0001)