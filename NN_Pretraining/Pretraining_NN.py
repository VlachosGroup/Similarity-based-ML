import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import pickle
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Function to count the number of weights in a layer
def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters())

# Function to create a random neural network within weight limits
def create_random_neural_network(input_size, max_weights, min_layers, max_layers, min_neurons, max_neurons):
    while True:
        layers = []
        num_layers = random.randint(min_layers, max_layers)  # Random number of layers
        total_weights = 0
        valid_network = True
        
        current_input_size = input_size
        
        for _ in range(num_layers):
            output_size = random.randint(min_neurons, max_neurons)
            layer = nn.Linear(current_input_size, output_size)
            
            # Count the weights of the current layer
            layer_weights = count_parameters(layer)
            
            # Check if adding this layer would exceed the weight limit
            if total_weights + layer_weights >= max_weights:
                valid_network = False
                break
            
            layers.append(layer)
            layers.append(nn.ReLU())
            
            # Update the total weights and input size for the next layer
            total_weights += layer_weights
            current_input_size = output_size
        
        # Add final output layer with 1 neuron
        output_layer = nn.Linear(current_input_size, 1)
        output_layer_weights = count_parameters(output_layer)
        
        # Check if adding the final output layer exceeds the weight limit
        if valid_network and total_weights + output_layer_weights <= max_weights:
            layers.append(output_layer)
            total_weights += output_layer_weights
            model = nn.Sequential(*layers)
            return model, total_weights

# Load data
x_qm9_cho = pd.read_csv('Data/descriptors_qm9_GDB17_CHO.csv')
qm9_cho = pd.read_csv('Data/qm9_GDB17_CHO.csv')
x_qm9_cho_shuffled = x_qm9_cho.sample(frac=1, random_state=42)
qm9_cho_shuffled = qm9_cho.sample(frac=1, random_state=42)

Cv_cho_shuffled_train = qm9_cho_shuffled.iloc[:30000]['C_v']
Cv_cho_shuffled_test = qm9_cho_shuffled.iloc[30000:45000]['C_v']
x_qm9_cho_shuffled_train = x_qm9_cho_shuffled.iloc[:30000]
x_qm9_cho_shuffled_test = x_qm9_cho_shuffled.iloc[30000:45000]

# Create a StandardScaler instance
scaler_Cv_shuffled = StandardScaler()
feats = ['NumAtoms', 'TIC1', 'NumRings', 'ATSC1Z', 'BCUTd-1l','BCUTdv-1l', 'exactmw', 'BalabanJ', 'ATS3Z', 'chi1v','n7FRing', 'VE1_A', 'Xch-3d', 'GATS1c', 'ATSC7d', 'ATSC6d', 'ATSC1dv', 'BertzCT', 'kappa3', 'n5Ring']
# Fit the scaler to the data and transform the data

X_qm9_cho_shuffled_train_scaled = scaler_Cv_shuffled.fit_transform(x_qm9_cho_shuffled_train[feats])
X_qm9_cho_shuffled_test_scaled = scaler_Cv_shuffled.transform(x_qm9_cho_shuffled_test[feats])

# Prepare tensors
x_train = torch.tensor(X_qm9_cho_shuffled_train_scaled, dtype=torch.float32)
y_train = torch.tensor(Cv_cho_shuffled_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output
x_test = torch.tensor(X_qm9_cho_shuffled_test_scaled, dtype=torch.float32)
y_test = torch.tensor(Cv_cho_shuffled_test.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output

input_size = x_train.shape[1]  # Adjusted input size based on data dimensions
max_weights = 10000
patience = 50  # Increased patience
n_iterations = 10000


models_1000 = [create_random_neural_network(input_size, max_weights, 2, 4, 10, 100)[0] for _ in range(1000)]
# Function to train a single model
def train_model(model_1000, x_train, y_train, x_test, y_test, n_iterations, patience):
   #  print(model_1000)
    initial_state_1000 = {k: v.clone() for k, v in model_1000.state_dict().items()}
    criterion_1000 = nn.MSELoss()
    optimizer_1000 = optim.Adam(model_1000.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_test_loss_1000 = float('inf')
    trigger_times_1000 = 0

    train_losses_1000 = []
    test_losses_1000 = []
    early_stopping_iterations = []

    for i in range(n_iterations):
     #   print(f"{i}th iteration")
        optimizer_1000.zero_grad()
        outputs = model_1000(x_train)
        loss = criterion_1000(outputs, y_train)
        loss.backward()
        optimizer_1000.step()
        
        train_losses_1000.append(loss.item())

        with torch.no_grad():
            test_outputs = model_1000(x_test)
            test_loss = criterion_1000(test_outputs, y_test)
            test_losses_1000.append(test_loss.item())
        
        if (i+1) % 100 == 0:
            logging.info(f'Iteration [{i+1}/{n_iterations}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
            sys.stdout.flush()  # Ensure the output is flushed

        if test_loss.item() < best_test_loss_1000:
            best_test_loss_1000 = test_loss.item()
            trigger_times_1000 = 0
        else:
            trigger_times_1000 += 1
            if trigger_times_1000 >= patience:
                logging.info(f'Early stopping at iteration {i+1}')
                early_stopping_iterations.append(i+1)
                sys.stdout.flush()  # Ensure the output is flushed
                break
    
    if len(early_stopping_iterations) == 0:
        early_stopping_iterations.append(n_iterations)

    trained_state_1000 = {k: v.clone() for k, v in model_1000.state_dict().items()}
    
    return {
        "initial_state": initial_state_1000,
        "trained_state": trained_state_1000,
        "train_losses": train_losses_1000,
        "test_losses": test_losses_1000,
        "best_test_loss": best_test_loss_1000,
        "early_stopping_iteration": early_stopping_iterations[-1]
    }

sample_number = 1 # do only the first 100 nns (for parallelizing), next sample would be 200:300 and so on

# Parallel training of models
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(train_model)(model_1000, x_train, y_train, x_test, y_test, n_iterations, patience) 
    for model_1000 in models_1000
)

# Collecting results
all_initial_states = []
all_trained_states = []
all_train_losses = []
all_test_losses = []
all_best_test_losses = []
all_early_stopping_iterations = []

for result in results:
    all_initial_states.append(result["initial_state"])
    all_trained_states.append(result["trained_state"])
    all_train_losses.append(result["train_losses"])
    all_test_losses.append(result["test_losses"])
    all_best_test_losses.append(result["best_test_loss"])
    all_early_stopping_iterations.append(result["early_stopping_iteration"])

# After training, save the collected data
all_data_1000 = {
    "initial_states": all_initial_states,
    "trained_states": all_trained_states,
    "train_losses": all_train_losses,
    "test_losses": all_test_losses,
    "best_test_losses": all_best_test_losses,
    "early_stopping_iterations": all_early_stopping_iterations
}

with open(f"result_Cv_nn_optimization_{sample_number}.pkl", "wb") as f:
    pickle.dump(all_data_1000, f)
