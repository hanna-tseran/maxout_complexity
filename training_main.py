from copy import deepcopy

import numpy as np
np.random.seed(123)
import torch
torch.manual_seed(123)
from torch import nn, optim
from torchvision import datasets, transforms

from helpers import generate_x
from init_method import init_params, InitMethod
from helpers import get_hidden_layer_sizes
from network import Network
from plotting import plot_regions_and_decision_boundary

PATH_TO_TRAINSET = 'mnist_train/'
PATH_TO_TESTSET = 'mnist_test/'
BATCH_SIZE = 128
ADAM_LR = 0.001

def create_linear_layer(fan_in, fan_out, init, zero_bias):
    new_layer = nn.Linear(fan_in, fan_out)
    weight, bias = init_params(init=init, K=1, fan_in=fan_in, fan_out=fan_out, zero_bias=zero_bias)
    new_layer.weight.data = nn.Parameter(torch.from_numpy(weight).float())
    new_layer.bias.data = nn.Parameter(torch.from_numpy(bias).float())
    return new_layer

class MaxoutLayer(nn.Module):
    def __init__(self, size_in, fan_out, K, init, zero_bias):
        super().__init__()
        self.size_in, self.fan_out = size_in, fan_out
        self.K = K
        weight, bias = init_params(init=init, K=self.K, fan_in=self.size_in, fan_out=self.fan_out, zero_bias=zero_bias)
        self.weight = nn.Parameter(torch.from_numpy(weight).float())
        self.bias = nn.Parameter(torch.from_numpy(bias).float())

    def forward(self, x):
        w_times_x = torch.transpose(torch.matmul(x, torch.transpose(self.weight, -2, -1)), 0, 1)
        w_times_x_plus_bias = torch.add(w_times_x, self.bias)
        result = torch.max(input=w_times_x_plus_bias, dim=-1)[0]
        return result

class MaxoutNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, K, init, zero_bias):
        super().__init__()

        self.activation = 'maxout'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.K = K
        self.init = init

        self.layers = []
        self.layers.append(MaxoutLayer(input_size, hidden_sizes[0], K, init, zero_bias=zero_bias))
        for hi in range(len(hidden_sizes) - 1):
            self.layers.append(MaxoutLayer(hidden_sizes[hi], hidden_sizes[hi + 1], K, init, zero_bias=zero_bias))
        self.layers.append(create_linear_layer(fan_in=hidden_sizes[-1], fan_out=output_size,
            init=InitMethod.HE_NORMAL, zero_bias=zero_bias))
        self.layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

def compute_accuracy(testloader, model):
    correct_count, all_count = 0, 0
    for images,labels in testloader:
      for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
    return correct_count / all_count

def compute_regions(model, axis_min, axis_max, points, axis_steps, labels, print_name):
    # First get the network weights and biases, and create an instance of our network class
    weights = []
    biases = []
    for layer_id, layer in enumerate(model.layers[:-1]):
        if type(layer) in [MaxoutLayer, nn.Linear]:
            weights.append(layer.weight.detach().numpy())
            biases.append(layer.bias.detach().numpy())

    net = Network(
        activation=model.activation,
        layer_sizes=([model.input_size] + model.hidden_sizes + [model.output_size]),
        init=model.init, # The weights and biases are copied, so it does not affect anything
        K=model.K,
        weights=deepcopy(weights),
        biases=deepcopy(biases))

    db_pieces, regions = net.db_and_regions_in_slice(points=points, axis_min=axis_min, axis_max=axis_max, labels=labels)

    print(f'{print_name}: {len(regions)} linear regions; {len(db_pieces)} linear pieces in the decision boundary')
    x_arr = generate_x(axis_min=axis_min, axis_max=axis_max, axis_steps=axis_steps)
    gradients = net.get_gradients(x_arr, points=points)
    plot_regions_and_decision_boundary(gradients=gradients.tolist(), axis_min=axis_min, axis_max=axis_max,
        axis_steps=axis_steps, points=points, db_pieces=db_pieces, print_name=print_name)

########################################################################################################################

def main():
    activation = 'maxout'
    input_size = 784
    output_size = 10
    num_neurons = 20
    depth = 3
    init = InitMethod.MAXOUT_HE_NORMAL
    K = 2
    axis_min = -50.
    axis_max = 50.
    axis_steps = 300
    zero_bias = False
    epochs = 10
    computation_stride = 5

    hidden_sizes = get_hidden_layer_sizes(num_neurons=num_neurons, depth=depth)

    if activation == 'maxout':
        model = MaxoutNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, K=K, init=init,
            zero_bias=zero_bias)
    else:
        raise Exception(f'Unknown activation {activation}')

    batch_num = 0
    points = []
    labels = None
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),])
    trainset = datasets.MNIST(PATH_TO_TRAINSET, download=True, train=True, transform=transform)
    testset = datasets.MNIST(PATH_TO_TESTSET, download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    dataiter = iter(trainloader)

    # Pick 3 points from 3 different labels
    pointloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    points = []
    point_labels = []
    for image, label in pointloader:
        l = label.detach().numpy()[0]
        if l not in point_labels:
            img = np.squeeze(image.view(image.shape[0], -1).detach().numpy())
            points.append(img)
            point_labels.append(l)
        if len(point_labels) == 3:
            break

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)
    batch_num = len(trainloader)

    compute_regions(model=model, axis_min=axis_min, axis_max=axis_max, points=points,
        labels=point_labels, axis_steps=axis_steps, print_name='before training')

    for e in range(epochs):
        running_loss = 0
        dataiter = iter(trainloader)
        for batch_id in range(batch_num):
            images, labels = next(dataiter)
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        accuracy = compute_accuracy(testloader, model)
        print(f'Epoch {e + 1}. Training loss: {running_loss / batch_num}. Accuracy: {accuracy}', flush=True)

        if (e + 1) % computation_stride == 0:
            compute_regions(model=model, axis_min=axis_min, axis_max=axis_max, points=points, labels=point_labels,
                axis_steps=axis_steps, print_name=f'after {e + 1} epochs')

if __name__ == '__main__':
    main()
