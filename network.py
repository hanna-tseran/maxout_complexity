import math
import warnings

import numpy as np
from scipy.linalg import orth
from scipy.optimize import linprog

from init_method import get_init_std, init_params, InitMethod
from region import Region, RegionFunction

def maxout(y):
    args = np.argmax(y, axis=1)
    return np.squeeze(np.take_along_axis(y, args.reshape(args.shape[0], 1, args.shape[1]), axis=1), axis=1), args

class Network:
    EPSILON = 1e-6
    SHARING_THRESHOLD = 2000

    def copy_weights_biases(self, activation_name, weights, biases):
        if activation_name == 'maxout':
            self.W = weights
            self.b = biases
        else:
            raise Exception(f'Wrong activation "{activation_name}"')

    def __init__(self, activation, layer_sizes, init, K=2, zero_bias=False, weights=None, biases=None):
        self.L = len(layer_sizes) - 1 # number of layers, input layer is not counted, its index is 0
        self.layer_sizes = np.array(layer_sizes, dtype=int) #n_in = n0 = 2
        self.n_in = self.layer_sizes[0]
        self.K = K # rank of the maxout
        if activation == 'maxout':
            self.activation = maxout
        else:
            raise Exception(f'"{activation}" is a wrong activation')
        self.init = init
        self.zero_bias = zero_bias

        if weights is not None:
            self.copy_weights_biases(activation_name=activation, weights=weights, biases=biases)
            return

        self.W = []
        self.b = []
        for l in range(self.L - 1):
            fan_in, fan_out = self.layer_sizes[l], self.layer_sizes[l + 1]
            weight, bias = init_params(init=self.init, K=self.K, fan_in=fan_in, fan_out=fan_out,
                zero_bias=self.zero_bias)
            self.W.append(weight)
            self.b.append(bias)
        # Add a linear layer on top
        fan_in, fan_out = self.layer_sizes[self.L - 1], self.layer_sizes[self.L]
        weight, bias = init_params(init=InitMethod.MAXOUT_HE_NORMAL, K=1, fan_in=fan_in, fan_out=fan_out,
            zero_bias=zero_bias)
        self.W.append(weight)
        self.b.append(bias)

    # Assumes normal distribution
    def estimate_c_bias(self):
        c_bias = 0.
        for l in range(self.L - 1):
            c_bias = max(c_bias,
                1 / (math.sqrt(2 * np.pi) * get_init_std(init=self.init, K=self.K, fan_in=self.layer_sizes[l])))
        return c_bias

    def estimate_c_grad(self, axis_min, axis_max, axis_steps):
        c_grad_arr = []
        grid = np.meshgrid(*[np.linspace(axis_min, axis_max, axis_steps) for _ in range(self.n_in)])
        x_arr = np.vstack(list(map(np.ravel, grid))).T

        # estimate C_grad for each neuron of the input
        num_neurons = np.sum(self.layer_sizes[1:-1])
        for x in x_arr:
            c_grad = 0
            input_grad = np.identity(self.layer_sizes[0])
            layer_input = x
            for l in range(0, self.L):

                layer_output = []
                output_grad = []
                for unit in range(0, self.layer_sizes[l + 1]):
                    if l < self.L - 1:
                        max_k = np.argmax([weight @ layer_input for weight in self.W[l][unit]])
                        weight = self.W[l][unit][max_k]
                    else:
                        weight = self.W[l][unit]

                    unit_weight = weight @ input_grad
                    output_grad.append(unit_weight)

                    unit_output = weight @ layer_input
                    layer_output.append(unit_output)

                    c_grad += np.linalg.norm(unit_weight)

                layer_input = layer_output
                input_grad = output_grad

            c_grad = c_grad / num_neurons
            c_grad_arr.append(c_grad)

        # For a better estimate a c_grad array should be averaged over several network instances
        # before taking the  maximum
        return np.max(c_grad_arr)

    # Auxilary function that checks if there is an intersection and returns a new regions if there is one
    def check_for_intersection(self, unitW, unitb, region, objective, input_bounds, n_in, k):
        try:
            featureW = unitW[k] @ region.function.W
            featureb = unitW[k] @ region.function.b + unitb[k]
        except ValueError as e:
            print(f'Exception in the intersection check: {e}')
            raise

        # Construct the set of inequalities that corresponds to the current preactivation feature attaining maximum
        lhs_inequalities = (region.lhs_inequalities + [unitW[j] @ region.function.W - featureW
            for j in range(self.K) if j != k])
        rhs_inequalities = (region.rhs_inequalities
            + [featureb - (unitW[j] @ region.function.b + unitb[j]) - self.EPSILON
            for j in range(self.K) if j != k])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    result = linprog(c=objective, A_ub=lhs_inequalities, b_ub=rhs_inequalities,
                    bounds=input_bounds, method='interior-point',
                    options={'sym_pos': False, 'lstsq': True, 'presolve': True})
                except Warning as w:
                    return None
        except Exception as e:
            return None

        if result.success:
            return Region(region.function, RegionFunction(region.next_layer_function.W + [featureW],
                region.next_layer_function.b + [featureb]), lhs_inequalities, rhs_inequalities)
        return None

    def check_for_equality(self, unitW, unitb, region, input_bounds, labels=None, n_in=None):
        if labels is None:
            labels = [0, 1]
        if n_in is None:
            n_in = self.layer_sizes[0]
        objective = np.zeros(n_in)

        pieces = []
        out_size = self.layer_sizes[-1]
        featureW = []
        featureb = []
        for i in range(out_size):
            featureW.append(unitW[i] @ region.function.W)
            featureb.append(unitW[i] @ region.function.b + unitb[i])

        A_eq_arr = []
        b_eq_arr = []
        lhs_inequalities_arr = []
        rhs_inequalities_arr = []
        for i_id, i in enumerate(labels):
            for j in labels[i_id + 1:]:
                A_eq_arr.append(np.asarray([featureW[j] - featureW[i]]))
                b_eq_arr.append(np.asarray([featureb[i] - featureb[j]]))

                l_ineq = []
                r_ineq = []
                for k in range(out_size):
                    if k != i and k != j and k in labels:
                        l_ineq.append(np.asarray(featureW[k] - featureW[i]))
                        r_ineq.append(np.asarray(featureb[i] - featureb[k]))
                lhs_inequalities_arr.append(l_ineq)
                rhs_inequalities_arr.append(r_ineq)

        for A_eq, b_eq, l_ineq, r_ineq in zip(A_eq_arr, b_eq_arr, lhs_inequalities_arr, rhs_inequalities_arr):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        result = linprog(c=objective,
                            A_ub=region.lhs_inequalities + l_ineq,
                            b_ub=region.rhs_inequalities + r_ineq,
                            A_eq=A_eq, b_eq=b_eq, bounds=input_bounds,
                            method='interior-point', options={'sym_pos': False, 'lstsq': True,
                            'presolve': True})
                    except Warning as w:
                        continue

            # This one usually tells that the problem is infeasible, which we are OK with
            except ValueError as e:
                continue

            if result.success:
                lhs_inequalities = region.lhs_inequalities + l_ineq
                rhs_inequalities = region.rhs_inequalities + r_ineq
                pieces.append(Region(region.function, None, lhs_inequalities, rhs_inequalities,
                    lhs_equalities=A_eq, rhs_equalities=b_eq))

        return pieces

    def count_decision_boundary_pieces(self, axis_min, axis_max, input_bounds=None, starting_region=None,
        objective=None, return_full=False, labels=None, n_in=None):

        regions = self.count_linear_regions_exactly(
            axis_min=axis_min,
            axis_max=axis_max,
            return_regions=True,
            input_bounds=input_bounds,
            starting_region=starting_region,
            objective=objective)

        pieces = []
        if n_in is None:
            n_in = self.layer_sizes[0]
        if input_bounds is None:
            input_bounds = np.repeat([[axis_min, axis_max]], n_in, axis=0)
        for region in regions:
            result = self.check_for_equality(unitW=self.W[-1], unitb=self.b[-1], region=region,
                input_bounds=input_bounds, labels=labels, n_in=n_in)
            for res in result:
                pieces.append(res)

        if return_full:
            return pieces, regions
        return len(pieces), len(regions)

    def count_linear_regions_exactly(self, axis_min, axis_max, input_bounds=None, starting_region=None, objective=None,
        return_regions=False):

        n_in = self.layer_sizes[0] # input dimension
        if objective is None:
            objective = np.zeros(n_in)

        # Construct the cube in which we will compute the results. We need this because by defualt
        # in linprog the bounds are (0, None), which does not work for us
        if input_bounds is None:
            input_bounds = np.repeat([[axis_min, axis_max]], n_in, axis=0)

        if starting_region is None:
            starting_region = Region(function=RegionFunction(np.identity(self.layer_sizes[0]),
                np.zeros(self.layer_sizes[0])), next_layer_function=RegionFunction(),
            lhs_inequalities=[], rhs_inequalities=[])
        linear_regions = [starting_region]

        all_new_regions = linear_regions
        new_local_regions = linear_regions
        for l in range(self.L - 1):
            for unit in range(self.layer_sizes[l+1]):
                local_regions = new_local_regions
                new_local_regions = []
                for region in local_regions:
                    for k in range(self.K):
                        new_region = self.check_for_intersection(
                            unitW=self.W[l][unit], unitb=self.b[l][unit], region=region,
                            objective=objective, input_bounds=input_bounds, n_in=n_in, k=k)
                        if new_region:
                            new_local_regions.append(new_region)
                if unit == self.layer_sizes[l+1] - 1:
                    for region in new_local_regions:
                        region.function = region.next_layer_function
                        region.next_layer_function = RegionFunction()

        all_new_regions = new_local_regions
        result = len(all_new_regions)
        if return_regions:
            result = all_new_regions

        return result

    def db_and_regions_in_slice(self, points, axis_min, axis_max, labels):
        p1, p2, p3 = points[0], points[1], points[2]
        input_bounds = np.asarray([[axis_min, axis_max] for _ in range(2)])
        origin = (p1 + p2 + p3) / 3.
        basis = orth(np.asarray([p2 - p1, p3 - p1]).T)
        starting_region = Region(function=RegionFunction(basis, origin), next_layer_function=RegionFunction())
        objective = np.zeros(2)

        return self.count_decision_boundary_pieces(
            axis_min=axis_min,
            axis_max=axis_max,
            input_bounds=input_bounds,
            starting_region=starting_region,
            objective=objective,
            return_full=True,
            labels=labels,
            n_in=2)

    def get_gradients(self, x, points=None):
        if points is not None:
            p1, p2, p3 = points[0], points[1], points[2]
            origin = (p1 + p2 + p3) / 3.
            basis = orth(np.asarray([p2 - p1, p3 - p1]).T)
            zx = ((basis @ x).T + origin).T
        else:
            zx = x

        num_samples = x.shape[1]
        db_mask = np.zeros(num_samples)
        for l in range(self.L):
            if l == self.L - 1:
                ggradl = np.repeat(np.expand_dims([self.W[l][0]], axis=0), num_samples, axis=0)
                grad = ggradl @ grad

            m = self.W[l] @ zx
            gradl = np.repeat(np.expand_dims(self.W[l], axis=0), num_samples, axis=0)

            # Last layer is linear, so there is no need to apply the activation function there
            if l < self.L - 1:
                zx = np.transpose(np.transpose(m, (2, 0, 1)) + self.b[l], (1, 2, 0))
                zx, argmaxes = self.activation(zx)
                argmaxes = np.repeat(np.expand_dims(argmaxes.T, axis=-1), self.layer_sizes[l], axis=-1)
                gradl = np.squeeze(np.take_along_axis(gradl, np.expand_dims(argmaxes, axis=-2), axis=-2), axis=-2)
                del argmaxes
            else:
                zx = (m.T + self.b[l]).T

            if l == 0:
                grad = gradl
            elif l < self.L - 1:
                grad = gradl @ grad
                del gradl

        return np.squeeze(np.sum(grad, axis=2))
