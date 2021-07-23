import torch
import torch.nn
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config
from model.net import SiameseNet
from model.losses import ContrastiveLoss
from model.data_loader import SiameseDataset
from scripts.utils import SummaryWriter

THRESHOLD = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SiameseDataset(training_csv=config.TESTFILE, train_dir=config.TESTDIR,
                         transform=transforms.Compose([transforms.Resize((105, 105)),
                                                      transforms.ToTensor()]))

test_data = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)

net = SiameseNet()

try:
    fpth = os.path.isfile(config.MODELFPATH)
except FileNotFoundError:
    print("File not found")

net.load_state_dict(torch.load(config.MODELFPATH))
net.to(device)
net.eval()
accuracy_counter = 0
for i, data in tqdm(enumerate(test_data, start=0)):
    x0, x1, label = data
    x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()

    op1, op2 = net(x0, x1)

    distance = F.pairwise_distance(op1, op2)

    if distance.item() < THRESHOLD:
        predicted_label = "Original"
    else:
        predicted_label = "Forgery"

    if label == torch.tensor([[0]]).cuda():
        actual_label = "Original"
    else:
        actual_label = "Forgery"

    if predicted_label == actual_label:
        accuracy_counter += 1

    print(f"distance from both specimens:{distance.item()}. Predicted Label: {predicted_label} and Actual Label: {actual_label}")

print(f"{accuracy_counter}/{i}; {accuracy_counter/i} accuracy")





