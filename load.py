import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def normalize_labels(y, y_mean, y_std): #### 标准化标签y
    y_normalized = (y - y_mean) / y_std
    return y_normalized

def restore_labels(y_normalized, y_mean, y_std): #### 恢复标签y的原始范围
    y_restored = y_normalized * y_std + y_mean
    return y_restored

def diabetes_data(test_size, random_state, valid_size, rescale=False): # 划分数据集并提取数据
    # 加载糖尿病数据集
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # 划分测试集
    X_train_0, X_test, y_train_0, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_0, y_train_0, test_size=valid_size, random_state=random_state)
    
    ### 提取训练集的标签y的平均值与标准差用于标准化数据集以及后续恢复原始标签范围 ###
    mean = y_train.mean()
    std = y_train.std()
    ### 可选：对数据集的标签进行标准化 ###
    if rescale:
        ### 对数据集的标签进行标准化 ###
        y_train = normalize_labels(y_train, mean, std)
        y_valid = normalize_labels(y_valid, mean, std)
        y_test = normalize_labels(y_test, mean, std)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, diabetes, mean, std

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(test_size, random_state, valid_size, batch_size, rescale=False):
    X_train, X_valid, X_test, y_train, y_valid, y_test, diabetes, mean, std = diabetes_data(test_size, random_state, valid_size, rescale)
    
    train_dataset = DiabetesDataset(X_train, y_train)
    valid_dataset = DiabetesDataset(X_valid, y_valid)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, mean, std #### 保留标签y的平均值与标准差用于后续恢复原始标签范围




if __name__ == "__main__":
    import pandas as pd
    
    X_train, X_valid, X_test, y_train, y_valid, y_test, diabetes, mean, std = diabetes_data(test_size=0.2, random_state=42, valid_size=0.2)
    X, y = diabetes.data, diabetes.target
    
    # 创建DataFrame便于查看
    feature_names = diabetes.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    # 查看数据
    print(f"特征字段: {diabetes.feature_names}")
    print(f"样本数: {len(df)}")
    print(f"特征数: {X.shape[1]}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"目标变量均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

    # 显示前5行数据
    print("\n数据示例（前5行）:")
    print(df.head())

    # 特征描述
    print("\n特征描述:")
    print("-" * 40)
    print("age: 年龄")
    print("sex: 性别")
    print("bmi: 身体质量指数")
    print("bp: 平均血压")
    print("s1~s6: 六种血清含量指标")
    print("-" * 40)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_valid.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    load_data(test_size=0.2, random_state=42, valid_size=0.2, batch_size=32, rescale=False)