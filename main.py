'''
Oren Gershony
Evaluate VAE on Chest X-ray Dataset and calculate statistics
'''
from data import load_data, inspect_data
from train import train
from VAE.VAE import VAE

def main() -> None:
    device = "cuda"
    # Dict with train, test, and val keys
    dataset = load_data()

    # Image shape is (32, 1, 128, 128)
    inspect_data(dataset)

    model = VAE().to(device)

    model, training_losses = train(model, dataset["train"], device)
    
    




if __name__ == "__main__":
    main()
