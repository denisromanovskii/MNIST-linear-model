import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(128, output)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_2(x)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MNIST_model = MyModel(784, 10).to(device)
loss_func = nn.CrossEntropyLoss() # функция потерь
opt = torch.optim.Adam(MNIST_model.parameters(), lr=0.001) # оптимизатор