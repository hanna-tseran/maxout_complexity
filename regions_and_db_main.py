import numpy as np
np.random.seed(123)

from formula import *
from helpers import generate_x, get_hidden_layer_sizes
from init_method import InitMethod
from network import Network
from plotting import plot_regions_and_decision_boundary

def main():
    activation = 'maxout'
    input_size = 2
    output_size = 2 # assume binary classification to count decision boundary pieces
    num_neurons = 20
    depth = 3
    init = InitMethod.MAXOUT_HE_NORMAL
    K = 2
    axis_min = -3.
    axis_max = 3.
    axis_steps = 300
    c_grad_axis_steps = 50
    zero_bias = False

    layer_sizes = ([input_size] + get_hidden_layer_sizes(num_neurons=num_neurons, depth=depth) + [output_size])
    net = Network(activation=activation, layer_sizes=layer_sizes, init=init, K=K, zero_bias=zero_bias)

    # Count activation regions and decision boundary pieces exactly
    db_pieces, regions = net.count_decision_boundary_pieces(axis_min=axis_min, axis_max=axis_max, return_full=True,
        n_in=input_size)
    print(f'Number of linear regions: {len(regions)}')
    print(f'Number of linear pieces in the decision boundary: {len(db_pieces)}')

    # Count linear regions approximaely
    x_arr = generate_x(axis_min=axis_min, axis_max=axis_max, axis_steps=axis_steps)
    gradients = net.get_gradients(x_arr)
    approx_regions_num = np.unique(gradients).shape[0]
    print(f'Approximate number of linear regions: {approx_regions_num}')

    # Theoretical estimates of the number of linear regions and linear pieces in the decision boundary
    print('#########################################')
    c_bias = net.estimate_c_bias()
    c_grad = net.estimate_c_grad(axis_min=axis_min, axis_max=axis_max, axis_steps=c_grad_axis_steps)
    predicted_num_regions = int(regions_formula(c_bias=c_bias, c_grad=c_grad, K=K, N=num_neurons, n_in=input_size,
        axis_min=axis_min, axis_max=axis_max))
    predicted_num_pieces = int(db_formula(c_bias=c_bias, c_grad=c_grad, K=K, N=num_neurons, n_in=input_size,
        axis_min=axis_min, axis_max=axis_max))
    print(f'Upper bound on the expected number using full formula. Number of linear regions: {predicted_num_regions}. '
        + f'Number of linear pieces in the decision boundary: {predicted_num_pieces}')

    predicted_num_regions = regions_predicted_growth_with_K(K=K, N=num_neurons, n_in=input_size)
    predicted_num_pieces = db_predicted_growth_with_K(K=K, N=num_neurons, n_in=input_size)
    print(f'Asymptotic with K. Number of linear regions: {predicted_num_regions}. '
        + f'Number of linear pieces in the decision boundary: {predicted_num_pieces}')

    predicted_num_regions = regions_predicted_growth(N=num_neurons, n_in=input_size)
    predicted_num_pieces = db_predicted_growth(N=num_neurons, n_in=input_size)
    print(f'Asymptotic without K. Number of linear regions: {predicted_num_regions}. '
        + f'Number of linear pieces in the decision boundary: {predicted_num_pieces}')

    # Plot linear regions and the decision boundary
    plot_regions_and_decision_boundary(gradients=gradients.tolist(), axis_min=axis_min, axis_max=axis_max,
        axis_steps=axis_steps, db_pieces=db_pieces, print_name='initialization')
    print('#########################################')
    print('Plotted regions and the decision boundary. The results are in the "images" folder.')

if __name__ == '__main__':
    main()
