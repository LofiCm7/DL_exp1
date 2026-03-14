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


if __name__ == "__main__":
    model1 = Basic_FNN_ReLU()
    model2 = Normal_FNN_ReLU()
    model3 = Deep_FNN_ReLU()
    
    print("Basic_FNN_ReLU:")
    print(model1)
    print("\nNormal_FNN_ReLU:")
    print(model2)
    print("\nDeep_FNN_ReLU:")
    print(model3)