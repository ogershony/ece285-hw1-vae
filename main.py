'''
Oren Gershony
Evaluate VAE on Chest X-ray Dataset and calculate statistics
'''
import json
import os

import torch
import torchvision.utils as vutils

from data import load_data, inspect_data
from train import train
from VAE.VAE import VAE
from eval_model import plot_loss, eval_model, generate_images

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Dict with train, test, and val keys
    dataset = load_data()

    # Image shape is (32, 1, 128, 128)
    inspect_data(dataset)

    model = VAE().to(device)

    model, training_losses = train(model, dataset["train"], device)
    torch.save(model.state_dict(), "vae.pt")

    os.makedirs("outputs", exist_ok=True)
    plot_loss(training_losses, "outputs/losses.png")
    eval_metrics = eval_model(model, dataset["test"], device)
    print(eval_metrics)
    generated = generate_images(model, device, num_images=16)

    # Persist everything the report needs
    vutils.save_image(generated, "outputs/generated_grid.png", nrow=8, padding=2)
    with open("outputs/results.json", "w") as f:
        json.dump({
            "metrics": {k: float(v) for k, v in eval_metrics.items()},
            "training_losses": training_losses,
            "num_train_images": len(dataset["train"].dataset),
            "num_test_images": len(dataset["test"].dataset),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        }, f, indent=2)

if __name__ == "__main__":
    main()
