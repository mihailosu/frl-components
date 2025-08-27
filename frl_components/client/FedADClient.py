import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from ..models import MemoryAutoencoder, Autoencoder
import tf_keras

from .training import train 

class FedADClient(fl.client.NumPyClient):
    def __init__(self, model=None, x_train=None, local_epochs=1, batch_size=5, learning_rate=0.001):
        self.model: MemoryAutoencoder | Autoencoder = model
        self.x_train = x_train

        self.local_epochs = local_epochs
        self.batch_size = batch_size

        '''
        NOTE: (to self) If the optimizer is instantiated within
        the fit function like it previously was, the optimizer loses
        track of the variables created by the model it seems, and cannot
        run backpropagation for some reason.
        '''
        self.optimizer = tf_keras.optimizers.Adam(learning_rate, weight_decay=1e-4)


    def get_parameters(self, config):

        model_weights = self.model.get_weights()

        return model_weights
    

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        train(self.model, 
              self.x_train, 
              epochs=self.local_epochs, 
              batch_size=self.batch_size, 
              optimizer=self.optimizer)

        return self.model.get_weights(), len(self.x_train), {}
