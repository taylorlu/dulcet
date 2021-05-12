import tensorflow as tf
import tensorflow_addons as tfa
from model.transformer_utils import positional_encoding, scaled_dot_product_attention
import numpy as np


class CNNResNorm(tf.keras.layers.Layer):
    def __init__(self,
                 out_size: int,
                 n_layers: int,
                 hidden_size: int,
                 kernel_size: int,
                 inner_activation: str,
                 last_activation: str,
                 padding: str,
                 normalization: str,
                 **kwargs):
        super(CNNResNorm, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.convolutions = [tf.keras.layers.Conv1D(filters=hidden_size,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for _ in range(n_layers - 1)]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=out_size,
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.last_activation = tf.keras.layers.Activation(last_activation)
        if normalization == 'layer':
            self.normalization = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(n_layers + 1)]
        elif normalization == 'batch':
            self.normalization = [tf.keras.layers.BatchNormalization() for _ in range(n_layers + 1)]
        else:
            assert False is True, f'normalization must be either "layer" or "batch", not {normalization}.'
    
    def call_convs(self, x, training):
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
            x = self.normalization[i](x, training=training)
        return x
    
    def call(self, inputs, training):
        x = self.call_convs(inputs, training=training)
        x = self.last_conv(x)
        x = self.last_activation(x)
        x = self.normalization[-2](x, training=training)
        return self.normalization[-1](inputs + x, training=training)


class CNNResNormWithIN(CNNResNorm):
    def __init__(self, *args, **kwargs):
        super(CNNResNormWithIN, self).__init__(*args, **kwargs)
        self.normalization_instance = [tfa.layers.InstanceNormalization(epsilon=1e-6) for _ in range(self.n_layers + 1)]

    def call_convs(self, x1, x2, training):
        for i in range(0, len(self.convolutions)):
            x1 = self.convolutions[i](x1)
            x1 = self.inner_activations[i](x1)
            x1 = self.normalization[i](x1, training=training)
            x2 = self.convolutions[i](x2)
            x2 = self.inner_activations[i](x2)
            x2 = self.normalization_instance[i](x2, training=training)
        return x1, x2
    
    def call(self, inputs1, inputs2, training):
        x1, x2 = self.call_convs(inputs1, inputs2, training=training)
        x1 = self.last_conv(x1)
        x1 = self.last_activation(x1)
        x1 = self.normalization[-2](x1, training=training)
        out1 = self.normalization[-1](inputs1 + x1, training=training)
        x2 = self.last_conv(x2)
        x2 = self.last_activation(x2)
        x2 = self.normalization_instance[-2](x2, training=training)
        out2 = self.normalization_instance[-1](inputs2 + x2, training=training)
        return out1, out2


class FFNResNorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(FFNResNorm, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units)
        self.activation = tf.keras.layers.Activation('relu')
        self.d2 = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ln = tf.keras.layers.BatchNormalization()
        self.last_ln = tf.keras.layers.BatchNormalization()
    
    def call(self, x, training):
        ffn_out = self.d1(x)
        ffn_out = self.d2(ffn_out)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.ln(ffn_out)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out, training=training)
        return self.last_ln(ffn_out + x)


class FFNResNormWithIN(FFNResNorm):
    def __init__(self, *args, **kwargs):
        super(FFNResNormWithIN, self).__init__(*args, **kwargs)
        self.ln_instance = tfa.layers.InstanceNormalization(epsilon=1e-6)
        self.last_ln_instance = tfa.layers.InstanceNormalization(epsilon=1e-6)

    def call(self, x1, x2, training):
        ffn_out1 = self.d1(x1)
        ffn_out1 = self.d2(ffn_out1)  # (batch_size, input_seq_len, model_dim)
        ffn_out1 = self.ln(ffn_out1)  # (batch_size, input_seq_len, model_dim)
        ffn_out1 = self.activation(ffn_out1)
        ffn_out1 = self.dropout(ffn_out1, training=training)
        
        ffn_out2 = self.d1(x2)
        ffn_out2 = self.d2(ffn_out2)  # (batch_size, input_seq_len, model_dim)
        ffn_out2 = self.ln_instance(ffn_out2)  # (batch_size, input_seq_len, model_dim)
        ffn_out2 = self.activation(ffn_out2)
        ffn_out2 = self.dropout(ffn_out2, training=training)
        return self.last_ln(ffn_out1 + x1), self.last_ln_instance(ffn_out2 + x2)


class HeadDrop(tf.keras.layers.Layer):
    """ Randomly drop n heads. """
    
    def __init__(self, **kwargs):
        super(HeadDrop, self).__init__(**kwargs)
    
    def call(self, batch, training: bool, drop_n_heads: int):
        if not training or (drop_n_heads == 0):
            return batch
        if len(tf.shape(batch)) != 4:
            raise Exception('attention values must be 4 dimensional')
        batch_size = tf.shape(batch)[0]
        head_n = tf.shape(batch)[1]
        if head_n == 1:
            return batch
        # assert drop_n_heads < head_n, 'drop_n_heads must less than number of heads'
        keep_head_batch = tf.TensorArray(tf.float32, size=batch_size)
        keep_mask = tf.concat([tf.ones(head_n - drop_n_heads), tf.zeros(drop_n_heads)], axis=0)
        for i in range(batch_size):
            t = tf.random.shuffle(keep_mask)
            keep_head_batch = keep_head_batch.write(i, t)
        keep_head_batch = keep_head_batch.stack()
        keep_head_batch = keep_head_batch[:, :, tf.newaxis, tf.newaxis]
        return batch * keep_head_batch * tf.cast(head_n / (head_n - drop_n_heads), tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, model_dim: int, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_drop = HeadDrop()
        
        assert model_dim % self.num_heads == 0
        
        self.depth = model_dim // self.num_heads
        
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        
        self.dense = tf.keras.layers.Dense(model_dim)
    
    def split_heads(self, x, batch_size: int):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q_in, mask, training, drop_n_heads):
        batch_size = tf.shape(q_in)[0]
        
        q = self.wq(q_in)  # (batch_size, seq_len, model_dim)
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = self.head_drop(scaled_attention, training=training, drop_n_heads=drop_n_heads)
        
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)
        concat_query = tf.concat([q_in, concat_attention], axis=-1)
        output = self.dense(concat_query)  # (batch_size, seq_len_q, model_dim)
        
        return output, attention_weights


class SelfAttentionResNorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionResNorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ln = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.last_ln = tf.keras.layers.BatchNormalization()
    
    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.mha(x, x, x, mask, training=training,
                                          drop_n_heads=drop_n_heads)  # (batch_size, input_seq_len, model_dim)
        attn_out = self.ln(attn_out)  # (batch_size, input_seq_len, model_dim)
        out = self.dropout(attn_out, training=training)
        return self.last_ln(out + x), attn_weights


class SelfAttentionResNormWithIN(SelfAttentionResNorm):

    def __init__(self, *args, **kwargs):
        super(SelfAttentionResNormWithIN, self).__init__(*args, **kwargs)
        self.ln_instance = tfa.layers.InstanceNormalization(epsilon=1e-6)
        self.last_ln_instance = tfa.layers.InstanceNormalization(epsilon=1e-6)
    
    def call(self, x1, x2, training, padding_mask, random_mask, drop_n_heads):
        attn_out1, attn_weights1 = self.mha(x1, x1, x1, random_mask, training=training,
                                          drop_n_heads=drop_n_heads)  # (batch_size, input_seq_len, model_dim)
        attn_out1 = self.ln(attn_out1)  # (batch_size, input_seq_len, model_dim)
        out1 = self.dropout(attn_out1, training=training)
        out1 = self.last_ln(out1 + x1)

        attn_out2, attn_weights2 = self.mha(x2, x2, x2, padding_mask, training=training,
                                          drop_n_heads=drop_n_heads)  # (batch_size, input_seq_len, model_dim)
        attn_out2 = self.ln_instance(attn_out2)  # (batch_size, input_seq_len, model_dim)
        out2 = self.dropout(attn_out2, training=training)
        out2 = self.last_ln_instance(out2 + x2)

        return out1, out2, attn_weights1, attn_weights2


class SelfAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)
    
    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads)
        return self.ffn(attn_out, training=training), attn_weights


class SelfAttentionDenseBlockWithIN(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionDenseBlockWithIN, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNormWithIN(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNormWithIN(model_dim, dense_hidden_units, dropout_rate=dropout_rate)
    
    def call(self, x1, x2, training, padding_mask, random_mask, drop_n_heads):
        attn_out1, attn_out2, attn_weights1, attn_weights2 = self.sarn(x1, x2, padding_mask=padding_mask, random_mask=random_mask, training=training, drop_n_heads=drop_n_heads)
        attn_out1, attn_out2 = self.ffn(attn_out1, attn_out2, training=training)
        return attn_out1, attn_out2, attn_weights1, attn_weights2


class SelfAttentionConvBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 conv_filters: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        super(SelfAttentionConvBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNorm(out_size=model_dim,
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding='same',
                               normalization='batch')
    
    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads)
        conv = self.conv(attn_out)
        return conv, attn_weights


class SelfAttentionConvBlockWithIN(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 conv_filters: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        super(SelfAttentionConvBlockWithIN, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNormWithIN(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNormWithIN(out_size=model_dim,
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding='same',
                               normalization='batch')
    
    def call(self, x1, x2, training, padding_mask, random_mask, drop_n_heads):
        attn_out1, attn_out2, attn_weights1, attn_weights2 = self.sarn(x1, x2, padding_mask=padding_mask, random_mask=random_mask, training=training, drop_n_heads=drop_n_heads)
        conv1, conv2 = self.conv(attn_out1, attn_out2)
        return conv1, conv2, attn_weights1, attn_weights2


class SelfAttentionBlocks(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 conv_filters: int,
                 dropout_rate: float,
                 dense_blocks: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        super(SelfAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.maximum_position_encoding = maximum_position_encoding
        self.pos_encoding_scalar = tf.Variable(1.)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_SADB = [
            SelfAttentionDenseBlockWithIN(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    dense_hidden_units=feed_forward_dimension, name=f'{self.name}_SADB_IN_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])]
        self.encoder_SACB = [
            SelfAttentionConvBlockWithIN(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                   name=f'{self.name}_SACB_IN_{i}', kernel_size=kernel_size,
                                   conv_activation=conv_activation, conv_filters=conv_filters)
            for i, n_heads in enumerate(num_heads[dense_blocks:])]
        self.attn_weight = tf.Variable(name='attn_weight', 
                                        dtype=tf.float32,
                                        validate_shape=True,
                                        initial_value=np.random.normal(size=[model_dim, 1]),
                                        trainable=True)
        self.spk_resnet = FFNResNorm(model_dim=model_dim,
                                     dense_hidden_units=model_dim,
                                     dropout_rate=dropout_rate)
        self.seq_resnet = FFNResNorm(model_dim=model_dim,
                                     dense_hidden_units=model_dim,
                                     dropout_rate=dropout_rate)
        self.spk_rnn = tf.keras.layers.GRU(self.model_dim, return_sequences=False)
        
    def call(self, inputs, training, fb_switch, padding_mask, min_index, random_padding_mask, drop_n_heads):
        shift_pos_encoding = positional_encoding(self.maximum_position_encoding, self.model_dim, start_index=min_index)
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x1 = x + self.pos_encoding_scalar * shift_pos_encoding[:, :seq_len, :]
        x2 = x + self.pos_encoding_scalar * self.pos_encoding[:, :seq_len, :]
        x1 = self.dropout(x1, training=training)
        x2 = self.dropout(x2, training=training)
        random_mask = tf.maximum(tf.cast(padding_mask, tf.float32), tf.cast(random_padding_mask[:, tf.newaxis, tf.newaxis, :], tf.float32))

        attention_weights = {}
        for i, block in enumerate(self.encoder_SACB):
            x1, x2, attn_weights1, attn_weights2 = block(x1, x2, training=training, padding_mask=padding_mask, random_mask=random_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_SelfAttention1'] = attn_weights1
            attention_weights[f'{self.name}_ConvBlock{i + 1}_SelfAttention2'] = attn_weights2
        for i, block in enumerate(self.encoder_SADB):
            x1, x2, attn_weights1, attn_weights2 = block(x1, x2, training=training, padding_mask=padding_mask, random_mask=random_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_SelfAttention1'] = attn_weights1
            attention_weights[f'{self.name}_DenseBlock{i + 1}_SelfAttention2'] = attn_weights2
        
        x1 = self.spk_resnet(x1)
        if(fb_switch<0.5):
            x1 = tf.reverse(x1, axis=[-1])
        x1 = self.spk_rnn(x1)
        x1 = tf.nn.l2_normalize(x1, 1)

        x2 = self.seq_resnet(x2)
        
        return x1, x2, attention_weights


class CrossAttentionResnorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionResnorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, q, k, v, training, mask):
        attn_values, attn_weights = self.mha(v, k=k, q_in=q, mask=mask, training=training)
        attn_values = self.dropout(attn_values, training=training)
        out = self.layernorm(attn_values + q)
        return out, attn_weights


class CrossAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training)
        
        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training)
        ffn_out = self.ffn(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionConvBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 conv_filters: list,
                 dropout_rate: float,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 **kwargs):
        super(CrossAttentionConvBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNorm(filters=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding=conv_padding,
                               dout_rate=dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training)
        
        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training)
        ffn_out = self.conv(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionBlocks(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.pos_encoding_scalar = tf.Variable(1.)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.CADB = [
            CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                     dense_hidden_units=feed_forward_dimension, name=f'{self.name}_CADB_{i}')
            for i, n_heads in enumerate(num_heads[:-1])]
        self.last_CADB = CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate,
                                                  num_heads=num_heads[-1],
                                                  dense_hidden_units=feed_forward_dimension,
                                                  name=f'{self.name}_CADB_last')
    
    def call(self, inputs, enc_output, training, decoder_padding_mask, encoder_padding_mask,
             reduction_factor=1):
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        attention_weights = {}
        for i, block in enumerate(self.CADB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_CrossAttention'] = attn_weights
        x, _, attn_weights = self.last_CADB(x, enc_output, training, decoder_padding_mask, encoder_padding_mask)
        attention_weights[f'{self.name}_LastBlock_CrossAttention'] = attn_weights
        return x, attention_weights


class DecoderPrenet(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(DecoderPrenet, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')  # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim, activation='relu')  # (batch_size, seq_len, model_dim)
        self.rate = tf.Variable(dropout_rate, trainable=False)
        self.dropout_1 = tf.keras.layers.Dropout(self.rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.rate)
    
    def call(self, x, training):
        self.dropout_1.rate = self.rate
        self.dropout_2.rate = self.rate
        x = self.d1(x)
        # use dropout also in inference for positional encoding relevance
        x = self.dropout_1(x, training=training)
        x = self.d2(x)
        x = self.dropout_2(x, training=training)
        return x


class Postnet(tf.keras.layers.Layer):
    
    def __init__(self, mel_channels: int, **kwargs):
        super(Postnet, self).__init__(**kwargs)
        self.mel_channels = mel_channels
        self.stop_linear = tf.keras.layers.Dense(3)
        self.mel_out = tf.keras.layers.Dense(mel_channels)
    
    def call(self, x):
        stop = self.stop_linear(x)
        mel = self.mel_out(x)
        return {
            'mel': mel,
            'stop_prob': stop,
        }


class StatPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 conv_filters: list,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 dense_activation: str,
                 dropout_rate: float,
                 **kwargs):
        super(StatPredictor, self).__init__(**kwargs)
        self.conv_blocks = CNNDropout(filters=conv_filters,
                                      kernel_size=kernel_size,
                                      padding=conv_padding,
                                      inner_activation=conv_activation,
                                      last_activation=conv_activation,
                                      dout_rate=dropout_rate)
        self.linear = tf.keras.layers.Dense(1, activation=dense_activation)
    
    def call(self, x, training, mask):
        x = x * mask
        x = self.conv_blocks(x, training=training)
        x = self.linear(x)
        return x * mask


class CNNDropout(tf.keras.layers.Layer):
    def __init__(self,
                 filters: list,
                 kernel_size: int,
                 inner_activation: str,
                 last_activation: str,
                 padding: str,
                 dout_rate: float):
        super(CNNDropout, self).__init__()
        self.n_layers = len(filters)
        self.convolutions = [tf.keras.layers.Conv1D(filters=f,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for f in filters[:-1]]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(self.n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=filters[-1],
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.last_activation = tf.keras.layers.Activation(last_activation)
        self.dropouts = [tf.keras.layers.Dropout(rate=dout_rate) for _ in range(self.n_layers)]
        self.normalization = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.n_layers)]
    
    def call_convs(self, x, training):
        for i in range(0, self.n_layers - 1):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
            x = self.normalization[i](x)
            x = self.dropouts[i](x, training=training)
        return x
    
    def call(self, inputs, training):
        x = self.call_convs(inputs, training=training)
        x = self.last_conv(x)
        x = self.last_activation(x)
        x = self.normalization[-1](x)
        x = self.dropouts[-1](x, training=training)
        return x


class Expand(tf.keras.layers.Layer):
    """ Expands a 3D tensor on its second axis given a list of dimensions.
        Tensor should be:
            batch_size, seq_len, dimension
        
        E.g:
        input = tf.Tensor([[[0.54710746 0.8943467 ]
                          [0.7140938  0.97968304]
                          [0.5347662  0.15213418]]], shape=(1, 3, 2), dtype=float32)
        dimensions = tf.Tensor([1 3 2], shape=(3,), dtype=int32)
        output = tf.Tensor([[[0.54710746 0.8943467 ]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.5347662  0.15213418]
                           [0.5347662  0.15213418]]], shape=(1, 6, 2), dtype=float32)
    """
    
    def __init__(self, model_dim, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.model_dimension = model_dim
    
    def call(self, x, dimensions):
        dimensions = tf.squeeze(dimensions, axis=-1)
        dimensions = tf.cast(tf.math.round(dimensions), tf.int32)
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        # build masks from dimensions
        max_dim = tf.math.reduce_max(dimensions)
        tot_dim = tf.math.reduce_sum(dimensions)
        index_masks = tf.RaggedTensor.from_row_lengths(tf.ones(tot_dim), tf.reshape(dimensions, [-1])).to_tensor()
        index_masks = tf.cast(tf.reshape(index_masks, (batch_size, seq_len * max_dim)), tf.float32)
        non_zeros = seq_len * max_dim - tf.reduce_sum(max_dim - dimensions, axis=1)
        # stack and mask
        tiled = tf.tile(x, [1, 1, max_dim])
        reshaped = tf.reshape(tiled, (batch_size, seq_len * max_dim, self.model_dimension))
        mask_reshape = tf.multiply(reshaped, index_masks[:, :, tf.newaxis])
        ragged = tf.RaggedTensor.from_row_lengths(mask_reshape[index_masks > 0], non_zeros)
        return ragged.to_tensor()
