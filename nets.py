import torch
import torch.nn as nn

class Basic_FNN_ReLU(nn.Module): # 基础ReLU_FNN 1层
    def __init__(self):
        super(Basic_FNN_ReLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

class Normal_FNN_ReLU(nn.Module): # 标准ReLU_FNN 2层
    def __init__(self):
        super(Normal_FNN_ReLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class Deep_FNN_ReLU(nn.Module): # 深层ReLU_FNN 4层
    def __init__(self):
        super(Deep_FNN_ReLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

class Normal_FNN_Sigmoid(nn.Module): # 标准Sigmoid_FNN 2层
    def __init__(self):
        super(Normal_FNN_Sigmoid, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class Normal_FNN_Tanh(nn.Module): # 标准Tanh_FNN 2层
    def __init__(self):
        super(Normal_FNN_Tanh, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class Normal_FNN_LeakyReLU(nn.Module): # 标准LeakyReLU_FNN 2层
    def __init__(self):
        super(Normal_FNN_LeakyReLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class Swish(nn.Module): # Swish激活函数
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Normal_FNN_Swish(nn.Module): # 标准Swish_FNN 2层
    def __init__(self):
        super(Normal_FNN_Swish, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            Swish(),
            nn.Linear(32, 16),
            Swish(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)




if __name__ == "__main__":
    model1 = Basic_FNN_ReLU()
    model2 = Normal_FNN_ReLU()
    model3 = Deep_FNN_ReLU()
    model4 = Normal_FNN_Sigmoid()
    model5 = Normal_FNN_Tanh()
    model6 = Normal_FNN_LeakyReLU()
    model7 = Normal_FNN_Swish()
    
    print("Basic_FNN_ReLU:")
    print(model1)
    print("\nNormal_FNN_ReLU:")
    print(model2)
    print("\nDeep_FNN_ReLU:")
    print(model3)
    print("\nNormal_FNN_Sigmoid:")
    print(model4)
    print("\nNormal_FNN_Tanh:")
    print(model5)
    print("\nNormal_FNN_LeakyReLU:")
    print(model6)
    print("\nNormal_FNN_Swish:")
    print(model7)