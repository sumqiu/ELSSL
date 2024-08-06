import os
import pickle
import numpy as np
import torch
from PIL import Image

# Define the path to the model file and the output directory for generated images
model_path = './network-snapshot.pkl'
outdir = './gen_outs/fake_imgs'
os.makedirs(outdir, exist_ok=True)  # Create the output directory if it doesn't exist
# Number of images to generate
n = 20000

# Load the model from the file
with open(model_path, 'rb') as f:
    # Assuming the model is saved under the key 'G_ema' in the pickle file
    G = pickle.load(f)['G_ema'].cuda()  # Move the model to GPU

# Generate images
for i in range(n):
    # Generate a random latent vector
    z = np.random.randn(1, G.z_dim)
    z = torch.Tensor(z).to('cuda')  # Convert to a PyTorch tensor and move to GPU

    # Class labels are not used in this example, so set to None
    c = None

    # Map the latent vector to the image space
    w = G.mapping(z, c)

    # Synthesize the image from the latent space
    img = G.synthesis(w)

    # Convert the image to the correct format and range
    # NCHW to NHWC, scale to [0, 255], and convert to uint8
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Convert the image tensor to a PIL Image and save it
    img = Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L')  # Take the first channel for grayscale
    img.save(os.path.join(outdir, f'{i:06d}.png'))  # Save the image with a zero-padded filename