
import tensorflow as tf

@tf.function
def reconstruction_loss(y_true, model_out, w=None, alpha=0.0002):
    """
    Calculate the reconstruction loss using squared L2 norm.
    
    Args:
    y_true (tensor): The ground truth values. Shape (batch_size, input_size)
    y_pred (tensor): The predicted values.
    w (tensor): Weight vectors for each row in the batch. (batch_size, encoding_size)
    alpha: Default value as described in 

    Returns:
    tensor: The mean squared L2 norm of the difference between y_true and y_pred.
    """

    #tr.print("Model out is: ", model_out)

    # Helper function calculating the entropy ot the weight vector
    def entropy(w_i):
        return tf.reduce_sum(-w_i * tf.math.log(w_i + 1e-10))
    
    # Calculate squared L2 norm for each row 
    # (each pair of values from the batch of the batch)
    l2_norm = tf.map_fn(
        tf.square, 
        tf.norm(y_true - model_out, ord=2, axis=-1)
    )

    if w is None:
        # Calculate standard loss only
        return tf.reduce_mean(l2_norm)


    # Calculate entropy of each weight vector in the batch
    entropy_out = tf.map_fn(entropy, w)

    #tr.print("entropy_out is: ", entropy_out)

    loss_terms = l2_norm + alpha * entropy_out

    #tr.print("loss terms are: ", loss_terms)

    errors = tf.reduce_mean(loss_terms)
    
    #tr.print("Errors #1 are: ", errors)

    return errors