import numpy as np
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err

curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        #evaluation part
        #Get input, compute the output of layer nodes.
        #this is multiplying each of the inputs by the weights and then summing up the values
        for i in range(self.num_nodes):
            self.layer_net[i] = np.dot(self.input_vals, np.transpose(self.weight[i])) + self.bias
            self.layer_out[i] = sig(self.layer_net[i])

    def backprop(self, other):
        #use backpropagation method to update weights
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                if self.layer_num == 1:
                    #this is taking the partial derivatives of each of the error functions with respect to each of the weights
                    update_input = LEARNING_RATE * other.weight_delta[0][i] * self.input_vals[j] * other.weight[0][i] * inv_sig(self.layer_out[i])
                    #weight[i][j] = the weight of input j to the current node i
                    self.weight[i][j] = self.weight[i][j] - update_input
                elif self.layer_num == 2:
                    self.weight_delta[i][j] = inv_sig(self.layer_out[i]) * inv_err(self.layer_out[i], other)
                    update_output = LEARNING_RATE * self.weight_delta[i][j] * self.input_vals[j]
                    self.weight[i][j] = self.weight[i][j] - update_output


class cfile:
    def __init__(self, name, mode = 'w'):
        self.file = open(name, mode)

    def w(self, string):
        self.file.writelines(str(string) + '\n')

    def close(self):
        self.file.close()
