import numpy as np

def get_hidden_layer_sizes(num_neurons, depth):
    neurons_per_layer = int(np.floor(num_neurons / depth))
    leftover_neurons = num_neurons - depth * neurons_per_layer
    hidden_layer_sizes = [neurons_per_layer for _ in range(depth)]
    while leftover_neurons > 0:
        for l in range(depth):
            hidden_layer_sizes[l] += 1
            leftover_neurons -= 1
            if leftover_neurons == 0:
                break
    return hidden_layer_sizes

def generate_x(axis_min, axis_max, axis_steps):
    grid_x, grid_y = np.meshgrid(np.linspace(axis_min, axis_max, axis_steps),
        np.linspace(axis_min, axis_max, axis_steps))
    return np.vstack([grid_x.ravel(), grid_y.ravel()])
