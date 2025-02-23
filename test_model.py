import torch
from tqdm import tqdm

from dataPreparation import test_loader, test_data
from MNISTmodel import MNIST_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = torch.load('MNIST-model-params.pt')
MNIST_model.load_state_dict(params)

test_loop = tqdm(test_loader, leave=False)
right_answers = 0
test_accurancy = 0

MNIST_model.eval()
for x, targets in test_loop:
    x = x.reshape(-1, 28 * 28).to(device)
    targets = targets.reshape(-1).to(torch.int32)
    targets = torch.eye(10)[targets].to(device)

    prediction = MNIST_model(x)

    right_answers += (prediction.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
    test_accurancy = right_answers / len(test_data)

    test_loop.set_description(f"Test-cases, test_accuracy={test_accurancy:.3f}")

print(f"Final accuracy={test_accurancy:.3f}")
