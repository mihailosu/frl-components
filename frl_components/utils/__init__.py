

import numpy as np
import random
import tensorflow as tf
import os


def set_random_seeds(seed_value = 42):
    
    # Set the seed value for scikit-learn's random number generator
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    random.seed(seed_value)

    # Set the seed value for NumPy's random number generator
    np.random.seed(seed_value)

    # Set the seed value for TensorFlow's random number generator
    tf.random.set_seed(seed_value)

    # random_state = check_random_state(seed_value)
