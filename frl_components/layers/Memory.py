import tensorflow as tf
# from tf_keras import layers
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class MemoryLayer(layers.Layer):

    def __init__(self, memory_dim, encoding_dim=None, lambda_cutoff=None, **kwargs):
        super(MemoryLayer, self).__init__(**kwargs)
        self.memory_dim = memory_dim
        self.lambda_cutoff = lambda_cutoff if lambda_cutoff else 1 / memory_dim
        self.encoding_dim = encoding_dim


    def build(self, input_shape):

        encoding_dim = self.encoding_dim if self.encoding_dim else input_shape[1]

        self.memory = self.add_weight(
            name='memory',
            shape=(self.memory_dim, encoding_dim),
            initializer='glorot_uniform',
            trainable=True)


    def call(self, inputs):
        '''
        
        Params:
            inputs -- (batch_size, encoding_dim)
        '''
        # First compute the cosine distances
        distances = self._compute_cosine_distance(inputs, self.memory)

        # Second, apply softmax per row to have weights sum to 1 for each row
        w = tf.nn.softmax(distances, axis=-1)

        # Now, apply the hard shrinkage operation based on the lambda cutoff
        w_shrunk = w - self.lambda_cutoff
        w = (tf.nn.relu(w_shrunk) * w) / (tf.abs(w_shrunk) + 1e-8)

        # Finally, renomalize the w tensor using L1 norm, which is still
        # (batch_size, memory_dim)
        # w = w / (tf.norm(w, ord=1) + 1e-8)
        w = w / (tf.norm(w, ord=1, axis=-1, keepdims=True) + 1e-8) # Must add keepdims to perform per-row divison


        # w x Memory
        # (batch_size, memory_dim) x (memory_dim, encoding_dim)
        output = tf.matmul(w, self.memory)

        # Return the outpus along with the weights used in the loss function
        return output, w


    @tf.function
    def _compute_cosine_distance(self, inputs, memory):
        '''
        Computes the cosine distance between the input vectors and the memory vectors.

        Params:
            inputs -- (batch_size, encoding_dim)
            memory -- (memory_dim, encoding_dim)

        Returns:
            (batch_size, memory_dim)
        '''
        # Normalizing the input and memory along the "encoding_dim" axis
        # ensures that their norms are 1. This allows us to skip the division
        # in the cosine similarity
        # cosine = tf.matmul(inp, mem) / (tf.norm(inp) * tf.norm(mem))
        # 
        # This effectively becomes just:
        # cosine = tf.matmul(inp, mem)
        inputs_normalized = tf.nn.l2_normalize(inputs, axis=-1)
        memory_normalized = tf.nn.l2_normalize(memory, axis=-1)

        # (batch_size, encoding_dim) x (memory_dim, encoding_dim) -- transpose B
        retval = tf.matmul(inputs_normalized, memory_normalized, transpose_b=True)

        # returns (batch_size, memory_dim)
        return retval


    def get_memory(self):
        return self.memory.numpy()
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "memory_dim": self.memory_dim,
            "encoding_dim": self.encoding_dim,
            "lambda_cutoff": self.lambda_cutoff,
        })
        return config
