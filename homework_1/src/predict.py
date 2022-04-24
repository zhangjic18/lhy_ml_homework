import csv
import torch
from torch.utils.data import DataLoader

from dataset import covid_dataset

test_dataloader = DataLoader(covid_dataset(flag="real_test"), batch_size=64, shuffle=False)

model = torch.load("../checkpoints/model.pth")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.to(device)

result = []

model.eval()
with torch.no_grad():
    for x, _ in test_dataloader:
        x = x.to(device)

        output = model(x).squeeze()

        predict = output * 7.61794257 + 16.43127986

        result += predict.cpu().numpy().tolist()

headers = ["id", "tested_positive"]

values = []
for index, item in enumerate(result):
    values.append({"id": index, "tested_positive": item})

with open("../data/result.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, headers)  # 写入表头的时候需要写入writerheader方法
    writer.writeheader()
    writer.writerows(values)
