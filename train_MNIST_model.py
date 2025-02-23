import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataPreparation import train_loader, val_loader, test_loader, train_data, val_data
from MNISTmodel import MNIST_model, loss_func, opt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 5
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(EPOCHS):
    MNIST_model.train()
    avg_train_loss = []
    right_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        x = x.reshape(-1, 28 * 28).to(device)

        targets = targets.reshape(-1).to(torch.int32)
        targets = torch.eye(10)[targets].to(device)

        # функция потери
        prediction = MNIST_model(x)
        lossing = loss_func(prediction, targets)

        # оптимизатор
        opt.zero_grad()
        lossing.backward()

        # градиентный спуск
        opt.step()

        avg_train_loss.append(lossing.item())
        mean_train_loss = sum(avg_train_loss) / len(avg_train_loss)

        right_answer += (prediction.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        train_loop.set_description(f"Train Epoch #{epoch+1}/{EPOCHS}, "
                                   f"train_loss={mean_train_loss:.3f}")

    train_accurancy = right_answer / len(train_data)
    train_loss.append(mean_train_loss)
    train_acc.append(train_accurancy)

    MNIST_model.eval()
    avg_val_loss = []
    true_answer = 0
    with torch.no_grad():
        val_loop = tqdm(val_loader, leave=False)
        for x, targets in val_loop:
            x = x.reshape(-1, 28 * 28).to(device)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)

            prediction = MNIST_model(x)
            lossing = loss_func(prediction, targets)

            avg_val_loss.append(lossing.item())
            mean_val_loss = sum(avg_val_loss) / len(avg_val_loss)

            true_answer += (prediction.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            val_loop.set_description(f"Vall Epoch #{epoch + 1}/{EPOCHS}, vall_loss={mean_val_loss:.3f}")

        val_accurancy = true_answer / len(val_data)

        val_loss.append(mean_val_loss)
        val_acc.append(val_accurancy)

    print(f"Epoch {epoch + 1}/{EPOCHS}, train_loss={mean_train_loss:.3f}, train_acc={train_accurancy:.3f},"
          f" val_loss={mean_val_loss:.3f}, val_acc={val_accurancy:.3f}")

torch.save(MNIST_model.state_dict(), 'MNIST-model-params.pt')






