import tensorflow as tf

from ..loss import reconstruction_loss

# NOTE: A custom training step is needed since the
# the reconstruction loss is a non-standard loss
# function
def train_step(model, optimizer, x):

    with tf.GradientTape() as tape:
        # Forward pass
        out = model(x, training=True)
        if isinstance(out, tuple):
            reconstructed, w = out
        else:
            reconstructed = out
            w = None
        # Compute loss
        loss = reconstruction_loss(x, reconstructed, w=w)

    tf.debugging.check_numerics(loss, "Loss contains NaN or Inf")
 
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for i, grad in enumerate(gradients):
        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
            print(f"NaN in gradient of variable {i}: {model.trainable_variables[i].name}")


    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


def train(model, dataset, optimizer=tf.keras.optimizers.SGD(0.001), epochs=10, batch_size=32):
    '''
    epochs defaults to 10 as per https://arxiv.org/pdf/2408.04442
    optimizer defaults to Adam as per 
    '''
    dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        for step, x in enumerate(dataset):
            loss = train_step(model, optimizer, x)
            epoch_loss_avg.update_state(loss)
            
        print(f"Epoch {epoch+1}: Average Loss: {epoch_loss_avg.result():.4f}")

