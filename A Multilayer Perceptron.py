#!/usr/bin/env python
# coding: utf-8

# In[9]:


# libraries

import numpy as np


# In[23]:


# MLP class with number of inputs layer and outputs
class MLP:
    
    def __init__(self, num_inputs=3, num_layer=[3,5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_layer = num_layer
        self.num_outputs = num_outputs
        
# get random weights for each connection
        self.weight_layer = []
        weight_list = [self.num_inputs] + self.num_layer + [self.num_outputs] # concatanate layers
        
        
        for i in range(len(weight_list)-1):
            w = np.random.rand(weight_list[i],weight_list[i+1])
            self.weight_layer.append(w)
    
# forward propagation calculations
    def forward_propagation(self, inputs):
        activation = inputs # the input is the first propagation
        for j in self.weight_layer:
            net_input = np.dot(activation, j)
            
            activation = self._sigmoid(net_input)
            
        return activation 
# sigmoid calculation
    def _sigmoid(self, x):
        return (1/(1+np.exp(-x)))


# In[25]:


input_ = 3
layer_ = [3,5]
output_ = 2

mlp = MLP()
inputs = np.random.rand(mlp.num_inputs)
outputs = mlp.forward_propagation(inputs)
print(outputs)
print(inputs)


# In[ ]:





# In[ ]:




