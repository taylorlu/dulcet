import tensorflow as tf
import math


def new_scaled_crossentropy(index=2, scaling=1.0):
    """
    Returns masked crossentropy with extra scaling:
    Scales the loss for given stop_index by stop_scaling
    """
    
    def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        padding_mask = tf.math.equal(targets, 0)
        padding_mask = tf.math.logical_not(padding_mask)
        padding_mask = tf.cast(padding_mask, dtype=tf.float32)
        stop_mask = tf.math.equal(targets, index)
        stop_mask = tf.cast(stop_mask, dtype=tf.float32) * (scaling - 1.)
        combined_mask = padding_mask + stop_mask
        loss = crossentropy(targets, logits, sample_weight=combined_mask)
        return loss
    
    return masked_crossentropy


def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss


def masked_mean_squared_error(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    mse = tf.keras.losses.MeanSquaredError()
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    mask = tf.reduce_max(mask, axis=-1)
    loss = mse(targets, logits, sample_weight=mask)
    return loss


def masked_mean_absolute_error(targets: tf.Tensor, logits: tf.Tensor, mask_value=0,
                               mask: tf.Tensor = None) -> tf.Tensor:
    mae = tf.keras.losses.MeanAbsoluteError()
    if mask is not None:
        mask = tf.math.logical_not(tf.math.equal(targets, mask_value))
        mask = tf.cast(mask, dtype=tf.int32)
        mask = tf.reduce_max(mask, axis=-1)
    loss = mae(targets, logits, sample_weight=mask)
    return loss


def masked_binary_crossentropy(targets: tf.Tensor, logits: tf.Tensor, mask_value=-1) -> tf.Tensor:
    bc = tf.keras.losses.BinaryCrossentropy(reduction='none')
    mask = tf.math.logical_not(tf.math.equal(logits,
                                             mask_value))  # TODO: masking based on the logits requires a masking layer. But masking layer produces 0. as outputs.
    # Need explicit masking
    mask = tf.cast(mask, dtype=tf.int32)
    loss_ = bc(targets, logits)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def ctc_loss(y_true, y_pred, label_length, input_length, blank=None):
    mask = tf.math.logical_not(tf.math.equal(label_length, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    return mask * tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits_time_major=False,
        blank_index=blank
    )


def amsoftmax_loss(targets: tf.Tensor, embeddings: tf.Tensor, weights: tf.Tensor, spk_count: int, s=50.0, m=0.5) -> tf.Tensor:
    weights = tf.nn.l2_normalize(weights, axis=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    cos_theta = tf.matmul(embeddings, weights)
    sin_theta = tf.sqrt(tf.subtract(1.0, tf.square(cos_theta)))
    cos_m_theta = s * tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m))

    threshold = math.cos(math.pi - m)

    cond_v = cos_theta - threshold
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
    keep_val = s*(cos_theta - m*sin_m)
    cos_m_theta_temp = tf.where(cond, cos_m_theta, keep_val)
    mask = tf.one_hot(targets, depth=spk_count)
    inv_mask = tf.subtract(1.0, mask)
    s_cos_theta = tf.multiply(s, cos_theta)
    logits = tf.add(tf.multiply(s_cos_theta, inv_mask), tf.multiply(cos_m_theta_temp, mask))

    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return cce(targets, logits)


def weighted_sum_losses(targets, pred, loss_functions, coeffs):
    total_loss = 0
    loss_vals = []
    for i in range(len(loss_functions)):
        loss = loss_functions[i](targets[i], pred[i])
        loss_vals.append(loss)
        total_loss += coeffs[i] * loss
    return total_loss, loss_vals
