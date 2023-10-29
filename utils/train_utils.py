import torch
from torch import nn


def Gtrain(train_loader, model, optimizer, device, criterion=nn.MSELoss()):
    """
    General training function for graph classification
    :param train_loader: DataLoader
    :param model: model
    :param optimizer: optimizer
    :param device: device
    :param criterion: loss function (default: MSELoss)
    """
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader, model, device, criterion=nn.L1Loss(reduction="mean")):
    """
    General test function for graph classification
    :param test_loader: DataLoader
    :param model: model
    :param device: device
    :param criterion: loss function (default: L1Loss)
    :return: error, accuracy
    """
    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(
                data.x,
                data.edge_index,
                data.batch,
            )

            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)
