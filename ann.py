from functions import functions, loss, init
import time
from graph import graph
import numpy as np
import sys
import csv

class Network:
    def __init__(self, layers, lrate, activation, init_method, loss_function='MSE', debug=False):
        '''
        layers: tuple of number of neurons (in integer) per layer
        lrate: learning rate
        activation: string name of activationg function
        init: Zero, Xavier, Xavier Normal, He, He Normal
        loss: objective function to be used (default: mean square error)
        '''
        if not isinstance(layers, tuple):
            raise TypeError('Wrong layer initialization!')
            return

        self.inputs = layers[0]
        self.lrate = lrate
        self.activation = functions[activation]
        self.loss_function_name = loss_function
        self.loss = loss[loss_function]
        self.init = init[init_method]
        self.debug = debug
        
        self.create_layer(layers)

    def create_layer(self, layers):
        self.layers = {
                'weights': [],
                'biases': [],
                'count': len(layers)-1
                }

        for prev, current in zip(layers, layers[1:]):
            self.layers['weights'].append(
                    np.array([
                        [self.init(prev, 1) for _ in range(prev)]
                        for _ in range(current)])
                    )
            self.layers['biases'].append([
                np.array([self.init(prev, 1)])
                for _ in range(current)])

    def train(self, in_file, out_file, max_epoch, min_error):
        inputs = self._load_file(in_file)
        targets = self._load_file(out_file)
        error = []                              # average error per epoch

        for epoch in range(max_epoch):
            error.append(0)

            for inp, target in zip(inputs, targets):
                prediction = self.predict(inp)
                error[-1] += self.loss['function']([(target, prediction)])
                self.backpropagate(inp, target)

            error[-1] /= len(targets)

            if error[-1] <= min_error:
                print(f'Finished training after {epoch} epochs, with error = {error[-1]}')
                break
        else:
            print(f'Reached maximum epoch without reaching the minimum error (error = {error[-1]}).')
        
        if self.debug:
            graph(x=error, figsize=(8, 6), title='Network training per epoch', legend=self.loss_function_name, xlabel='epoch', ylabel='error')

    def predict(self, inputs):
        y = np.atleast_2d(inputs).T

        for layer_w, layer_b in zip(self.layers['weights'], self.layers['biases']):
            z = np.dot(layer_w, y) + layer_b
            y = self.activation['function'](z)

        return y.flatten()

    def backpropagate(self, inputs, target):
        if self.layers['count'] > 1:
            print('Multi-layer not implemented yet.')
            sys.exit(0)

        x = np.atleast_2d(inputs).T
        x_hat  = np.dot(self.layers['weights'][0], x) + self.layers['biases'][0]
        y_pred = self.activation['function'](x_hat)

        delta1 = self.loss['derivative']([(target, y_pred)])                        # ∂E/∂(y-pred)
        #print(f'{target=}, {y_pred=}')
        #time.sleep(1)
        delta2 = self.activation['derivative'](x_hat)                               # ∂(y-pred)/∂(x-hat)
        delta3 = inputs                                                             # ∂(x-hat)/∂W

        grad_weights = np.dot(delta1 * delta2, x.T)                                 # ∇F(W) = ∂E/∂W
        self.layers['weights'][0] -= self.lrate * grad_weights                      # gradient descent
        self.layers['biases'][0] -= self.lrate * delta1 * delta2                    # gradient descent

    @staticmethod
    def _load_file(filename):
        with open(filename, mode='r') as file:
            return [np.array([float(value) for value in row])
                    for row in csv.reader(file)]
