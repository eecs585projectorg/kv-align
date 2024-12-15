# Ref: https://github.com/karpathy/build-nanogpt, StreamingLLM, ChatGPT
# Ref: https://docs.ray.io/en/latest/tune/examples/includes/async_hyperband_example.html, https://docs.ray.io/en/latest/tune/getting-started.html, https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import warnings
import time
from ray import tune
from ray import init as ray_init
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import tempfile

save_path = "/extra_space" # Modify this as appropriate

os.makedirs(f'{save_path}/temp', exist_ok=True)
os.makedirs(f'{save_path}/ray', exist_ok=True)

tempfile.tempdir = f'{save_path}/temp' # Ref: https://stackoverflow.com/a/78297621
ray_init(_temp_dir=f"{save_path}/ray") # Ref: https://discuss.ray.io/t/set-root-temporary-path-with-ray-tune/8862/2

warnings.simplefilter('ignore', category=FutureWarning)

key = False
if key:
    target = "keys"
else:
    target = "values"

num_blocks = 6
num_heads = 12
cache_size = 256
start_size = 4
sequence_dim = cache_size - start_size - 1
embed_dim = 64

def load_shard(index, start_size, sequence_dim):
    with torch.no_grad():
        # Load keys or values
        initial_target_path = f"training_data/misaligned_{target}_shard{index}.pt"
        new_target_path = f"training_data/aligned_{target}_shard{index}.pt"
        print("Loading from", initial_target_path, "and", new_target_path)
        initial_target = torch.load(initial_target_path).detach()
        new_target = torch.load(new_target_path).detach()

        num_examples = initial_target.size(0)
        X = initial_target[:, :, :, start_size + 1 :, :].reshape(-1, 64)

        print("Shape of Input X", X.shape)
        
        # Create position, block, and head tensors
        P, B, H = [], [], []
        for a in range(num_examples):
            for b in range(num_blocks):
                for c in range(num_heads):
                    for d in range(sequence_dim):
                        P.append(d)
                        B.append(b)
                        H.append(c)
        P = torch.tensor(P)
        B = torch.tensor(B)
        H = torch.tensor(H)

        # Reshape Y to match
        Y = new_target[:, :, :, start_size:, :].reshape(-1, 64)
        
        assert X.shape == Y.shape
    return TensorDataset(X, P, B, H, Y)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, num_pos=256, num_blocks=6, num_heads=12, emb_pos_dim=32, emb_blocks_dim=4, emb_heads_dim=8, l1=128, l2=128):
        super().__init__()

        self.emb_pos = nn.Embedding(num_pos, emb_pos_dim)
        self.emb_blocks = nn.Embedding(num_blocks, emb_blocks_dim)
        self.emb_heads = nn.Embedding(num_heads, emb_heads_dim)
        
        # Simple MLP for continuous variables X and Y
        self.fc1 = nn.Linear(64 + emb_pos_dim + emb_blocks_dim + emb_heads_dim, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 64)

        
    def forward(self, x, p, b, h):
        # Forward pass for continuous features and categorical variable
        p_emb = self.emb_pos(p)
        b_emb = self.emb_blocks(b)
        h_emb = self.emb_heads(h)
        x_z = torch.cat([x, p_emb, b_emb, h_emb], dim=1)  # Combine continuous and embedded categorical data
        x_z = torch.relu(self.fc1(x_z))
        x_z = torch.relu(self.fc2(x_z))
        x = self.fc3(x_z) + x
        return x


def train_keys_values(config):
    model = NeuralNetwork(l1=config["l1"], l2=config["l2"]).to("cuda")
    criterion = nn.MSELoss()  
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    num_epochs = 5
    log_interval = 10  

    batch_size = int(config["batch_size"])
    val_batch_size = 8192
    val_length = 10

    print("Training started")
    print("batch_size", batch_size)
 
    num_shards = 5

    # Variables for logging loss values
    train_losses = []
    val_losses = []
    train_times = []
    val_times = []

    # Training loop

    total_batch_count = 0

    for epoch in range(num_epochs):
        for shard_index in range(1, num_shards + 1):

            dataset = load_shard(shard_index, start_size, sequence_dim)

            # Perform a single split into training and validation datasets (80% train, 20% validation)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

            print("len(train_loader)", len(train_loader), "len(val_loader)", len(val_loader))

            for batch_idx, (batch_X, batch_P, batch_B, batch_H, batch_Y) in enumerate(train_loader):

                total_batch_count += 1

                model.train()
                optimizer.zero_grad()

                start_time = time.time()
                
                # Move data to the GPU
                batch_X, batch_P, batch_B, batch_H, batch_Y = batch_X.to("cuda"), batch_P.to("cuda"), batch_B.to("cuda"), batch_H.to("cuda"), batch_Y.to("cuda")
                
                # Forward pass (both continuous and categorical variables)
                predictions = model(batch_X, batch_P, batch_B, batch_H)
                
                # Calculate loss
                loss = criterion(predictions, batch_Y)
                loss.backward()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # torch.cuda.synchronize()
                end_time = time.time()
                
                # Log loss every log_interval batches
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Total Count: {total_batch_count}, Training Loss: {loss.item():.4f}, Norm: {norm:.4f}")
                    train_losses.append(loss.item())
                    train_times.append(total_batch_count)
                
            model.eval()  # Switch to evaluation mode to compute validation loss
            with torch.no_grad():
                val_loss = 0
                start_time = time.time()
                for val_idx, (val_batch_X, val_batch_P, val_batch_B, val_batch_H, val_batch_Y) in enumerate(val_loader):
                    if val_idx + 1 == val_length:
                        break
                    val_batch_X, val_batch_P, val_batch_B, val_batch_H, val_batch_Y = val_batch_X.to("cuda"), val_batch_P.to("cuda"), val_batch_B.to("cuda"), val_batch_H.to("cuda"), val_batch_Y.to("cuda")
                    val_predictions = model(val_batch_X, val_batch_P, val_batch_B, val_batch_H)
                    val_loss += criterion(val_predictions, val_batch_Y).item()
                torch.cuda.synchronize()
                end_time = time.time()
                time_for_val = end_time - start_time
                val_loss /= val_length  # Average the validation loss over all validation batches
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Total Count: {total_batch_count}, Validation Loss: {val_loss:.4f}, Time for Validation: {time_for_val:.4f}")
                val_losses.append(val_loss)
                val_times.append(total_batch_count)

                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    checkpoint = None
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),  # Save model weights
                            'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
                            'train_losses': train_losses,      # Optionally, save losses if you want to resume training stats
                            'train_times': train_times,         # Optionally, save other metrics like time
                            'val_losses': val_losses,
                            'val_times': val_times
                        }, os.path.join(temp_checkpoint_dir, "model.pth")
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(
                        {"loss": val_loss}, checkpoint=checkpoint
                    )

search_space = {
    "l1": tune.choice([2 ** i for i in range(5, 9)]),
    "l2": tune.choice([2 ** i for i in range(5, 9)]),
    "batch_size": tune.choice([2 ** i for i in range(5, 15)]),
    "lr": tune.loguniform(1e-4, 1e-1),
}


tuner = tune.Tuner(
    tune.with_resources(train_keys_values, resources={"gpu": 1}),
    run_config=train.RunConfig(
        name="key_value_tune",
        storage_path="/{dir_name}",
    ),
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=ASHAScheduler(metric="loss", mode="min", max_t=15),
    ),
    param_space=search_space,
)

results = tuner.fit()

best_trial = results.get_best_result("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.metrics['loss']}")


