import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import covid_dataset
from utils import AverageMeter


def main(args):
    total_dataset = covid_dataset(flag="train")

    dataset_len_list = [math.floor(0.8 * len(total_dataset)),
                        math.floor(0.1 * len(total_dataset)),
                        math.floor(0.1 * len(total_dataset)),
                        ]
    dataset_len_list[0] += (len(total_dataset) - sum(dataset_len_list))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_dataset, dataset_len_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = nn.Sequential(nn.Linear(93, 64, bias=True),
                          nn.ReLU(inplace=True),
                          nn.Linear(64, 1, bias=True),
                          )

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.MSELoss(reduction="mean")
    if args.optimizer == "SGDm":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(40, args.epoch, 40)),
                                                         gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)

    train(args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler)

    test_mse = test(model, test_dataloader, criterion, device)

    print("========================================test========================================")
    print("test mse: {:.6f}".format(test_mse))


def train(args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler):
    print("========================================train========================================")
    model.train()

    epoch_loss = AverageMeter()
    best_val_epoch = -1
    min_val_mse = float("inf")

    for epoch in range(args.epoch):
        epoch_loss.reset()

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x).squeeze()

            loss = criterion(output, y)

            assert loss.item() != np.nan

            epoch_loss.update(loss.item(), x.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = test(model, val_dataloader, criterion, device)
        if val_mse < min_val_mse:
            torch.save(model, "../checkpoints/model.pth")
            min_val_mse = val_mse
            best_val_epoch = epoch

        if args.scheduler == "MultiStepLR":
            scheduler.step()
        else:
            scheduler.step(val_mse)

        print("epoch: {}/{}, train mse: {:.6f}, val mse: {:.6f}, lr: {:.10f}".format(
            epoch, args.epoch, epoch_loss.avg, val_mse, optimizer.param_groups[0]['lr']))

    print("========================================best val========================================")
    print("best_val_epoch: {}, min_val_mse: {:.6f}".format(best_val_epoch, min_val_mse))


def test(model, test_dataloader, criterion, device):
    model.eval()

    with torch.no_grad():
        total_loss = AverageMeter()

        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x).squeeze()

            loss = criterion(output, y)

            total_loss.update(loss.item(), x.shape[0])

    return total_loss.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVID-19 Cases Prediction")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=270)
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--optimizer", type=str, default="SGDm")
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau")
    parser.add_argument("--device", type=str, default="4")

    args = parser.parse_args()
    main(args=args)
