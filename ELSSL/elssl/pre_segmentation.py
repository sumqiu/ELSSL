import cv2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from network.u_net import Unet
from dataloader.dataset import LiverDataset
import torch
from torchvision.transforms import transforms


def pre_seg(model_path, res_img_path, seg_path):   # Parameters are the path to the model and the path to save the image

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    model = Unet(1, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    liver_dataset = LiverDataset(seg_path, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    plt.ion()
    i = 1
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            # Assume y is a PyTorch tensor
            # First, ensure the tensor is of float type for subsequent normalization operations
            img_tensor = torch.squeeze(y).float()
            # Calculate the maximum value of the tensor
            max_value = torch.max(img_tensor)
            # Normalize the tensor to the [0, 255] range and convert to uint8 type
            # Note: There is no need to call .byte() here, as using .float() has already ensured the data type
            img_tensor = (255 * (img_tensor / max_value)).type(torch.uint8)
            # Convert the PyTorch tensor to a NumPy array
            img_y = img_tensor.numpy()
            # Check the number of channels in the image, if it is 1, remove the single channel dimension
            if img_y.ndim == 3 and img_y.shape[0] == 1:
                img_y = img_y[0].copy()
            # Save the NumPy array as an image file
            cv2.imwrite(res_img_path + '/' + str(i) + '.png', img_y)
            i += 1
