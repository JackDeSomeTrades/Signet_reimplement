import torch.nn
import torch.optim as optimizer
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.utils.data import DataLoader

import config
from model.net import SiameseNet
from model.losses import ContrastiveLoss
from model.data_loader import SiameseDataset
from scripts.utils import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(comment=config.run_name)  # pvnr

dataset = SiameseDataset(training_csv=config.TRAINFILE, train_dir=config.TRAINDIR,
                         transform=transforms.Compose([transforms.Resize((105, 105)),
                                                      transforms.ToTensor()]))

train_data = DataLoader(dataset, shuffle=True, num_workers=8, batch_size=config.batch_size)

net = SiameseNet().cuda()
loss = ContrastiveLoss()
optim = optimizer.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=0.0005)


def train():
    l = []
    counter = []

    for epoch in tqdm(range(config.EPOCHS), desc='Epoch:'):
        for index, data in tqdm(enumerate(train_data, start=0), desc='train data iteration:'):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optim.zero_grad()
            op1, op2 = net(img0, img1)

            cntr_loss_pairwise = loss(op1, op2, label)
            cntr_loss_pairwise.backward()
            optim.step()

            l.append(cntr_loss_pairwise.item())
            # writer.add_scalar("loss", cntr_loss_pairwise.item(), index)
        writer.add_scalar("loss", l[-1], epoch)
        
    writer.flush()
    writer.close()

    return net


print(f"Model training with the following parameters - {config.EPOCHS} epochs on {device}")

model = train()
torch.save(model.state_dict(), config.MODELFPATH)
print("Training complete. Model saved.")



