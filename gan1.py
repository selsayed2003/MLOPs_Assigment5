import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import mlflow
import mlflow.pytorch

print("--- Script Started ---")

# --- MLflow Pillar: Set Experiment Name ---
mlflow.set_experiment("Assignment3_SarahElsayed")

# Networks
class Discriminator(nn.Module):
    def __init__(self, image_dim=784):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim=64, image_dim=784):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x)

# Evaluation Function (Modified to return values for MLflow logging)
def evaluate_gan(disc, gen, loader, device, z_dim):
    disc.eval()
    gen.eval()
    real_correct, fake_correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, _ in loader:
            batch_size = images.size(0)
            images = images.view(-1, 784).to(device)
            
            # Test on Real
            real_out = disc(images).view(-1)
            real_correct += (real_out > 0.5).sum().item()
            
            # Test on Fake
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = gen(noise)
            fake_out = disc(fake_images).view(-1)
            fake_correct += (fake_out < 0.5).sum().item() 
            
            total += batch_size
            
    return 100 * real_correct / total, 100 * fake_correct / total

# --- Main Training Function ---
def train_gan(run_name, lr, batch_size, epochs=10, z_dim=64):
    # --- MLflow Pillar: Wrap in start_run ---
    with mlflow.start_run(run_name=run_name):
        
        # --- MLflow Pillar: Log Parameters ---
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("z_dim", z_dim)
        
        # --- MLflow Pillar: Add Tags ---
        mlflow.set_tag("student_id", "Sarah_Elsayed")
        mlflow.set_tag("model_architecture", "GAN")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dim = 28 * 28 * 1
        
        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        disc = Discriminator(image_dim).to(device)
        gen = Generator(z_dim, image_dim).to(device)
        
        opt_disc = optim.Adam(disc.parameters(), lr=lr)
        opt_gen = optim.Adam(gen.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print(f"\nStarting {run_name} | LR: {lr} | Batch: {batch_size}")

        for epoch in range(epochs):
            disc.train()
            gen.train()
            
            epoch_lossD = 0.0
            epoch_lossG = 0.0
            
            for batch_idx, (real, _) in enumerate(loader):
                real = real.view(-1, 784).to(device)
                curr_batch_size = real.shape[0]

                # 1. Train Discriminator
                noise = torch.randn(curr_batch_size, z_dim).to(device)
                fake = gen(noise)
                
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real)) 
                
                disc_fake = disc(fake.detach()).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward()
                opt_disc.step()

                # 2. Train Generator
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output)) 
                
                gen.zero_grad()
                lossG.backward()
                opt_gen.step()
                
                epoch_lossD += lossD.item()
                epoch_lossG += lossG.item()

            # Average losses for the epoch
            avg_lossD = epoch_lossD / len(loader)
            avg_lossG = epoch_lossG / len(loader)
            
            # Evaluate to get accuracies
            acc_real, acc_fake = evaluate_gan(disc, gen, test_loader, device, z_dim)

            # --- MLflow Pillar: Live Logging ---
            mlflow.log_metric("loss_Discriminator", avg_lossD, step=epoch)
            mlflow.log_metric("loss_Generator", avg_lossG, step=epoch)
            mlflow.log_metric("accuracy_Real", acc_real, step=epoch)
            mlflow.log_metric("accuracy_Fake", acc_fake, step=epoch)

            print(f"  Epoch [{epoch+1}/{epochs}] Loss D: {avg_lossD:.4f}, Loss G: {avg_lossG:.4f} | Acc Real: {acc_real:.1f}%")

        # --- MLflow Pillar: Save Model Flavor ---
        # We save both the generator and discriminator so the full GAN is versioned
        # --- MLflow Pillar: Save Model Flavor ---
        mlflow.pytorch.log_model(gen, "generator_model")
        mlflow.pytorch.log_model(disc, "discriminator_model")
        
        # NEW: Save the Run ID to a text file for GitHub Actions to read
        current_run_id = mlflow.active_run().info.run_id
        with open("model_info.txt", "w") as f:
            f.write(current_run_id)
            
        print(f"Run {run_name} complete and saved to MLflow. Run ID: {current_run_id}")

# --- Phase 3: Execute 5 Runs ---
# Note: I reduced epochs to 5 for speed, you can bump it back to 20 if you have time.
experiments = [
    {"name": "Run1_Baseline",  "lr": 3e-4, "batch_size": 32},
    {"name": "Run2_High_LR",   "lr": 1e-2, "batch_size": 32}, # Expect mode collapse/instability
    {"name": "Run3_Low_LR",    "lr": 1e-5, "batch_size": 32}, # Expect slow convergence
    {"name": "Run4_Large_Batch","lr": 3e-4, "batch_size": 256},
    {"name": "Run5_Small_Batch","lr": 3e-4, "batch_size": 8}
]

for exp in experiments:
    train_gan(run_name=exp["name"], lr=exp["lr"], batch_size=exp["batch_size"], epochs=5)

print("\nAll runs complete! Go to http://localhost:5000 to view your results.")