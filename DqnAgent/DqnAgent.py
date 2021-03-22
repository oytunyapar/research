import torch
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_dqn_agent_torch(layer_sizes):
    model_layers = []
    for layer_iterator in range(len(layer_sizes) - 1):
        model_layers.append(torch.nn.Linear(layer_sizes[layer_iterator], layer_sizes[layer_iterator + 1]))
        if layer_iterator == len(layer_sizes) - 2:
            model_layers.append(torch.nn.Softmax(dim=0))
        else:
            model_layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*model_layers)


def create_dqn_agent_tensorflow(layer_sizes):
    model = Sequential()
    initializer = tensorflow.keras.initializers.GlorotUniform()
    for layer_iterator in range(len(layer_sizes) - 1):
        if layer_iterator == len(layer_sizes) - 2:
            model.add(Dense(layer_sizes[layer_iterator + 1], activation='softmax',
                            input_dim=layer_sizes[layer_iterator],
                            name=str(layer_iterator),
                            kernel_initializer=initializer))
            break

        model.add(Dense(layer_sizes[layer_iterator + 1], activation='relu', input_dim=layer_sizes[layer_iterator],
                        name=str(layer_iterator), kernel_initializer=initializer))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
