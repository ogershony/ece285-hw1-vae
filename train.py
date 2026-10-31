import torch 
from torch.optim import Adam
import torch.nn.functional as F

def vae_loss(pred_image, image, mean, variance):
    decoder_loss = F.binary_cross_entropy(pred_image, image, reduction="sum")
    encoder_loss = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())

    return decoder_loss + encoder_loss

def train(model, dataloader, device):
    lr = 1e-3
    num_epochs = 100

    training_losses = []
    optimizer = Adam(model.parameters(), lr=lr)

    for i in range(num_epochs):
        total_loss = 0
        for idx, (image, label) in enumerate(dataloader):
            image = image.to(device)

            optimizer.zero_grad()

            pred_image, mean, variance = model(image)

            loss = vae_loss(pred_image, image, mean, variance)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        training_losses.append(total_loss)

    return model, training_losses

        
