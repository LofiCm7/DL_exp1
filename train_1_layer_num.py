import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from load import load_data, restore_labels
from nets import Normal_FNN_ReLU, Basic_FNN_ReLU, Deep_FNN_ReLU

# ***************************************************
# 网络深度的影响
#
# 实验设置：
# 固定学习率: 0.001
# 固定激活函数: ReLU
# 设计3种网络结构：
#
# | 网络类型 | 隐藏层数 | 网络类名称 |
# |---------|---------|------------------|
# | 浅层网络 | 1个隐藏层 | Basic_FNN_ReLU  |
# | 中等网络 | 2个隐藏层 | Normal_FNN_ReLU |
# | 深层网络 | 4个隐藏层 | Deep_FNN_ReLU   |
# ***************************************************

def train_model(which_model):
    # 1.超参数设置与设备选择
    EPOCHS = 200          # 训练总轮数
    BATCH_SIZE = 32       # 批次大小
    LEARNING_RATE = 0.005 # 学习率
    
    # 自动检测并使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}\n")

    # 2.加载数据
    train_loader, valid_loader, test_loader, mean, std = load_data(
        test_size=0.2, 
        random_state=42, 
        valid_size=0.2, 
        batch_size=BATCH_SIZE, 
        rescale_test=False #不标准化测试集
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
    print("开始训练...")
    
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
        
        #验证
        model.eval()
        valid_loss_sum = 0.0
        
        # 验证阶段不需要计算梯度，节约显存和算力
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                valid_loss_sum += loss.item() * batch_X.size(0)
                
        # 计算平均验证 Loss
        avg_valid_loss = valid_loss_sum / len(valid_loader.dataset)
        
        # 保存表现最好的模型权重
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), "./model_weights/best_diabetes_model.pth")
        
        # 每 20 轮打印一次进度
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

    print("已将最佳模型保存为 'best_diabetes_model.pth'")
    print("-" * 40)

    # ==========================================
    # 5. 测试集评估 (还原为真实物理量度)
    # ==========================================
    print("\n加载最佳模型进行测试集评估...")
    # 加载刚刚保存的最佳权重
    model.load_state_dict(torch.load("best_diabetes_model.pth", weights_only=True))
    model.eval()
    
    absolute_errors = []
    
    with torch.no_grad():
        for batch_X, batch_y_true in test_loader:
            # 此时 batch_y_true 是未经标准化的真实糖尿病指数（一百多或两百多）
            batch_X = batch_X.to(device)
            
            # 模型输出的是标准化空间下的预测值（0 左右）
            normalized_preds = model(batch_X)
            
            # 把预测值搬回 CPU 并转为 numpy 数组进行处理（或者用 torch 的方法也可以）
            normalized_preds = normalized_preds.cpu()
            
            # 【核心逻辑】：使用 load.py 里的 restore_labels 把预测值还原为真实指标！
            real_preds = restore_labels(normalized_preds, mean, std)
            
            # 计算绝对误差 (|预测值 - 真实值|)
            batch_errors = torch.abs(real_preds - batch_y_true)
            absolute_errors.extend(batch_errors.numpy().flatten())
            
    # 计算平均绝对误差 (Mean Absolute Error, MAE)
    mae = np.mean(absolute_errors)
    print(f"测试集评估完成！")
    print(f"模型的真实预测误差 (MAE): 平均偏离真实病情指数 {mae:.2f} 分")


if __name__ == "__main__":
    train_model()