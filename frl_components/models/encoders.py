
import tensorflow as tf
from tf_keras import layers, Sequential, Model
from ..layers.Memory import MemoryLayer
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class MemoryAutoencoder(Model):

    def __init__(self, memory_dim=None, encoding_dim=None, lambda_cutoff=None, layer_strat='power', **kwargs):
        super(MemoryAutoencoder, self).__init__()

        self.memory_dim = memory_dim if memory_dim else 32
        self.lambda_cutoff = lambda_cutoff
        self.encoding_dim = encoding_dim
        self.layer_strat = layer_strat

    def build(self, input_shape):

        self.input_dim = input_shape[-1]

        if self.layer_strat == 'power':
            layer_dims = self._power_of_two_ranges(self.encoding_dim, self.input_dim)
        elif self.layer_strat == 'half':
            layer_dims = self._halving_ranges(self.encoding_dim, self.input_dim)
        elif isinstance(self.layer_strat, list):
            layer_dims = self.layer_strat
        else:
            layer_dims = []

        # Append input dim to end of layer_dims
        layer_dims.append(self.input_dim)

        # Append encoding_dim to beginning of layer_dims
        layer_dims.insert(0, self.encoding_dim)

        input_layers = [
            layers.Dense(dim, activation='relu', input_shape=(dim,)) for dim in reversed(layer_dims)
        ]

        self.encoder = Sequential(input_layers)

        encoding_dim = self.encoding_dim if self.encoding_dim else input_shape[1]
        
        self.memory = MemoryLayer(
            self.memory_dim, 
            encoding_dim=encoding_dim, 
            lambda_cutoff=self.lambda_cutoff
        )

        self.decoder = Sequential([
            layers.Dense(dim, activation='relu') for dim in layer_dims
        ])


    def _power_of_two_ranges(self, min, max):
        ranges = []
        current = 2
        while current <= max:
            if current > min:
                ranges.append(current)
            current = current * 2

        return ranges


    def _halving_ranges(self, min, max):
        '''
        Generates layer dimensions by dividing the number of neruons
        in each subsequent layer by 2.
        '''
        ranges = []
        current = max // 2
        while current > min:
            ranges.append(current)
            current = current // 2
        
        ranges.reverse()
        
        return ranges

    def call(self, inputs):
        encoded = self.encoder(inputs)

        z, w = self.memory(encoded)
        
        decoded = self.decoder(z)
        
        return decoded, w
    

    def _compute_distances_to_memory(self, encoded):
        distances = []
        for row in self.memory:
            distances.append(self._cosine(encoded, row))
        return tf.stack([distances])
    

    def _cosine(self, v1, v2):
        return tf.reduce_sum(v1 * v2) / (tf.norm(v1) * tf.norm(v2))


    def encode(self, inputs):
        return self.encoder(inputs)


    def decode(self, inputs):
        return self.decoder(inputs)
    

    def set_memory(self, memory):
        self.memory.set_memory(memory)

    def get_memory(self):
        return self.memory.get_memory()
    
    def get_config(self):
        base_config = super().get_config()
        custom_config = {
            "memory_dim": self.memory_dim,
            "encoding_dim": self.encoding_dim,
            "lambda_cutoff": self.lambda_cutoff,
            "layer_strat": self.layer_strat,
        }
        return {**base_config, **custom_config}

    
