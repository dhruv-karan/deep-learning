# -*- coding: utf-8 -*-
import numpy as np
timesteps =100
input_features =32
output_features = 64

inputs = np.random.random((timesteps,input_features))

state_t = np.zeros((output_features))

W = np.random.random((output_features,input_features))
U = np.random.random((output_features,output_features))
b = np.random.random((output_features))

succ_output = []

for i in inputs:
    print(i.shape)
    output_t = np.tanh(np.dot(W,i)+np.dot(U,state_t)+b)
    succ_output.append(output_t)
    state_t = output_t

final_output_seq = np.concatenate(succ_output,axis=0)

