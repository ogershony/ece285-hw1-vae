'''
This file contains helper functions for eval, such as loss visualization, 
inception and FID score, also generates example images as well '''

import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless backend (no display on cluster nodes)
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel

def plot_loss(training_losses, output_name = "losses.png"):
    ''' Plot training losses '''
    
    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label='Training Loss', color='blue', linewidth=2)
    
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Loss')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the plot
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free up memory


def _prep_for_inception(images):
    ''' InceptionV3 expects 3-channel 299x299 input: repeat the grayscale channel and upsample '''
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return images


def eval_model(model, test_dataset, device="cuda", batch_size=32):
    ''' Uses pytorch ignite inception and FID score to evaluate model against the test dataset '''
    model.eval()
    
    # Initialize PyTorch Ignite metrics
    # Standard FID uses the 2048-d pool features of Inception-v3. Ignite's default FID() instead
    # uses the 1000-d softmax class probabilities, which gives tiny, non-comparable values (~0.1).
    fid_metric = FID(num_features=2048,
                     feature_extractor=InceptionModel(return_features=True, device=device),
                     device=device)
    is_metric = InceptionScore(device=device)
    
    fid_metric.reset()
    is_metric.reset()
    
    with torch.no_grad():
        for batch in test_dataset:
            real_images = batch[0]
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            
            z = torch.randn(b_size, 128, device=device)

            fake_images = model.decoder(z)

            fake_images, real_images = _prep_for_inception(fake_images), _prep_for_inception(real_images)

            fid_metric.update((fake_images, real_images))
            is_metric.update(fake_images)
            
    # Compute the final scores
    fid_score = fid_metric.compute()
    inception_score = is_metric.compute()
    
    return {
        "FID": fid_score,
        "InceptionScore": inception_score
    }

def generate_images(model, device="cuda", output_dir = "images", num_images = 10):
    ''' Generates 10 random images using the model '''
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(num_images, 128, device=device)

        generated_tensor = model.decoder(z)

    for i, img_tensor in enumerate(generated_tensor):
        file_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
        
        # save_image automatically handles scaling and saving tensors as images.
        # normalize=True is helpful if your VAE decoder uses tanh activations (ranging from -1 to 1).
        vutils.save_image(img_tensor, file_path, normalize=True)
            
    return generated_tensor
