import os
import argparse
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from network.u_net import Unet
from dataloader.dataset import LiverDataset
from dataloader.process import *
import torch


def train_model(model, criterion, optimizer, dataload, num_epochs=20, model_dir='model'):
    # Ensure that the model save directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_weight_filename = None  # Filename for tracking the best model weights

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))

        # Save the last round of model weights
        if epoch == num_epochs - 1:
            best_weight_filename = os.path.join(model_dir, 'best_weight.pth')
            torch.save(model.state_dict(), best_weight_filename)

    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    return model, best_weight_filename


# Train a U-Net model
def pre_train10(output_dir, batch_size, num_epochs, train_path):
    # Parameter parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    args = parser.parse_args()

    # data preprocessing
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # creating output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = Unet(1, 1).to(device)
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(train_path, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    train_model(model, criterion, optimizer, dataloaders, args.num_epochs, output_dir)


# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    for i in range(1, 11):   # Sequentially train 10 models
        print("Model " + str(i) + " is being trained")
        out_path = "models/m" + str(i)
        pre_train10(out_path, train_batch_size(i), train_num_epochs(), train_path(i))
        print("***************************************************************************")
        print("***************************************************************************")