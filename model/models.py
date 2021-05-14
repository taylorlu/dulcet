import tensorflow as tf
import numpy as np
import random

from model.transformer_utils import create_mel_padding_mask, create_mel_random_padding_mask
from utils.losses import weighted_sum_losses, masked_mean_absolute_error, ctc_loss, amsoftmax_loss
from data.text import TextToTokens
from model.layers import StatPredictor, Expand, SelfAttentionBlocks, FFNResNorm


class ASREncoder(tf.keras.models.Model):
    
    def __init__(self,
                 english_lexicon_path,
                 pinyin_lexicon_path,
                 mel_channels: int,
                 spk_count: int,
                 encoder_model_dimension: int,
                 encoder_num_heads: list,
                 encoder_maximum_position_encoding: int,
                 encoder_prenet_dimension: int,
                 dropout_rate: float,
                 encoder_dense_blocks: int,
                 encoder_attention_conv_filters: int = None,
                 encoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 debug=False,
                 **kwargs):
        super(ASREncoder, self).__init__(**kwargs)
        self.spk_count = spk_count
        self.drop_n_heads = 0
        self.text_pipeline = TextToTokens.default(english_lexicon_path, 
                                                  pinyin_lexicon_path, 
                                                  add_start_end=False)
        self.vocab_size = self.text_pipeline.tokenizer.vocab_size
        self.encoder_prenet = tf.keras.layers.Dense(encoder_prenet_dimension, 
                                                    name='encoder_prenet')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_maximum_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Encoder')
        self.classifier = tf.keras.layers.Dense(self.vocab_size)
        self.amsoftmax_weights = tf.Variable(name='amsoftmax_weights', 
                                            dtype=tf.float32,
                                            validate_shape=True,
                                            initial_value=np.random.normal(size=[encoder_model_dimension, spk_count]),
                                            trainable=True)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32)
        ]
        self.encoder_signature = [
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32)
        ]
        self.debug = debug
        self._apply_all_signatures()
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
        self.forward_encoder = self._apply_signature(self._forward_encoder, self.encoder_signature)
    
    def _call_encoder(self, inputs, training):
        min_index, padding_mask, random_padding_mask = create_mel_random_padding_mask(inputs)
        fb_switch = tf.random.uniform(shape=[], maxval=1, seed=random.randint(0, 2147483647), dtype=tf.float32)
        enc_input = self.encoder_prenet(inputs)
        spk_output, enc_output, attn_weights = self.encoder(enc_input,
                                                            training=training,
                                                            fb_switch=fb_switch,
                                                            padding_mask=padding_mask,
                                                            min_index=min_index,
                                                            random_padding_mask=random_padding_mask,
                                                            drop_n_heads=self.drop_n_heads)
        enc_output = self.classifier(enc_output,
                                     training=training)
        return spk_output, enc_output, padding_mask, attn_weights
    
    def _forward(self, inp):
        model_out = self.__call__(inputs=inp,
                                  training=False)
        return model_out
    
    def _forward_encoder(self, inputs):
        return self._call_encoder(inputs, training=False)
    
    def _gta_forward(self, spk, mel_inp, phon_tar, mel_inp_len, phon_tar_len, training):
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=mel_inp,
                                      training=training)
            phon_loss = tf.reduce_mean(self.loss[0](phon_tar, model_out['encoder_output'], phon_tar_len, mel_inp_len))
            spk_loss = self.loss[1](spk, model_out['spk_output'], self.amsoftmax_weights, self.spk_count)
            loss = self.loss_weights[0] * phon_loss + self.loss_weights[1] * spk_loss

        model_out.update({'loss': loss})
        model_out.update({'losses': {'spk_loss': spk_loss, 'phon_loss': phon_loss}})
        return model_out, tape
    
    def _train_step(self, spk, mel_inp, phon_tar, mel_inp_len, phon_tar_len):
        model_out, tape = self._gta_forward(spk, mel_inp, phon_tar, mel_inp_len, phon_tar_len, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _val_step(self, spk, mel_inp, phon_tar, mel_inp_len, phon_tar_len):
        model_out, _ = self._gta_forward(spk, mel_inp, phon_tar, mel_inp_len, phon_tar_len, training=False)
        return model_out
    
    def _compile(self, optimizer):
        self.loss_weights = [1., 1.]
        self.compile(loss=[ctc_loss,
                           amsoftmax_loss],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def call(self, inputs, training):
        spk_output, encoder_output, padding_mask, encoder_attention = self._call_encoder(inputs, training)
        model_out = {}
        model_out.update({'encoder_attention': encoder_attention, 'encoder_output': encoder_output, 'text_mask': padding_mask})
        model_out.update({'spk_output': spk_output})
        return model_out
    
    def predict(self, mel_inp):
        out_dict = {}
        spk_output, encoder_output, padding_mask, encoder_attention = self.forward_encoder(mel_inp)
        out_dict.update({'encoder_attention': encoder_attention, 'encoder_output': encoder_output, 'text_mask': padding_mask})
        out_dict.update({'spk_output': spk_output})
        return out_dict
    
    def set_constants(self,
                      learning_rate: float = None):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
    
    def decode_phoneme(self, phoneme):
        return self.text_pipeline.tokenizer.decode(phoneme)


class ForwardTransformer(tf.keras.models.Model):
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 dropout_rate: float,
                 decoder_num_heads: list,
                 encoder_num_heads: list,
                 encoder_maximum_position_encoding: int,
                 decoder_maximum_position_encoding: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 duration_conv_filters: list,
                 duration_kernel_size: int,
                 predictors_dropout: float,
                 mel_channels: int,
                 with_stress: bool,
                 encoder_attention_conv_filters: list = None,
                 decoder_attention_conv_filters: list = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 debug=False,
                 **kwargs):
        super(ForwardTransformer, self).__init__(**kwargs)
        self.text_pipeline = TextToTokens.default(add_start_end=False,
                                                  with_stress=with_stress)
        self.mel_channels = mel_channels
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_model_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_maximum_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Encoder')
        self.dur_pred = StatPredictor(conv_filters=duration_conv_filters,
                                      kernel_size=duration_kernel_size,
                                      conv_padding='same',
                                      conv_activation='relu',
                                      dense_activation='relu',
                                      dropout_rate=predictors_dropout,
                                      name='dur_pred')
        self.expand = Expand(name='expand', model_dim=encoder_model_dimension)
        self.speaker_fc = tf.keras.layers.Dense(encoder_model_dimension, name="speaker_fc")
        self.decoder = SelfAttentionBlocks(model_dim=decoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=decoder_num_heads,
                                           feed_forward_dimension=decoder_feed_forward_dimension,
                                           maximum_position_encoding=decoder_maximum_position_encoding,
                                           dense_blocks=decoder_dense_blocks,
                                           conv_filters=decoder_attention_conv_filters,
                                           kernel_size=decoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Decoder')
        self.out = tf.keras.layers.Dense(mel_channels)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 512), dtype=tf.float32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
        self.forward_masked_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ]
        self.debug = debug
        self._apply_all_signatures()
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
    
    def _train_step(self, input_sequence, target_sequence, target_durations, spk_emb):
        target_durations = tf.expand_dims(target_durations, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        with tf.GradientTape() as tape:
            model_out = self.__call__(input_sequence, target_durations=target_durations, spk_emb=spk_emb, training=True)
            loss, loss_vals = weighted_sum_losses((target_sequence,
                                                   target_durations),
                                                  (model_out['mel'][:, :mel_len, :],
                                                   model_out['duration']),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _compile(self, optimizer):
        self.loss_weights = [1., 1.]
        self.compile(loss=[masked_mean_absolute_error,
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def _val_step(self, input_sequence, target_sequence, target_durations, spk_emb):
        target_durations = tf.expand_dims(target_durations, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        model_out = self.__call__(input_sequence, target_durations=target_durations, spk_emb=spk_emb, training=False)
        loss, loss_vals = weighted_sum_losses((target_sequence,
                                               target_durations),
                                              (model_out['mel'][:, :mel_len, :],
                                               model_out['duration']),
                                              self.loss,
                                              self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        return model_out
    
    def _forward(self, input_sequence, spk_emb, durations_scalar):
        return self.__call__(input_sequence, target_durations=None, spk_emb=spk_emb, training=False,
                             durations_scalar=durations_scalar, max_durations_mask=None,
                             min_durations_mask=None)
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def call(self, x, target_durations, spk_emb, training, durations_scalar=1., max_durations_mask=None,
             min_durations_mask=None):
        encoder_padding_mask = create_encoder_padding_mask(x)
        x = self.encoder_prenet(x)
        x, encoder_attention = self.encoder(x, training=training, padding_mask=encoder_padding_mask)
        padding_mask = 1. - tf.squeeze(encoder_padding_mask, axis=(1, 2))[:, :, None]
        spk_emb = tf.math.softplus(self.speaker_fc(spk_emb))
        spk_emb = tf.expand_dims(spk_emb, 1)
        x = x + spk_emb #tf.tile(pitch_embed, [1, tf.shape(x)[1], 1])

        durations = self.dur_pred(x, training=training, mask=padding_mask)

        if target_durations is not None:
            use_durations = target_durations
        else:
            use_durations = durations * durations_scalar
        if max_durations_mask is not None:
            use_durations = tf.math.minimum(use_durations, tf.expand_dims(max_durations_mask, -1))
        if min_durations_mask is not None:
            use_durations = tf.math.maximum(use_durations, tf.expand_dims(min_durations_mask, -1))
        mels = self.expand(x, use_durations)
        expanded_mask = create_mel_padding_mask(mels)
        mels, decoder_attention = self.decoder(mels, training=training, padding_mask=expanded_mask, reduction_factor=1)
        mels = self.out(mels)
        model_out = {'mel': mels,
                     'duration': durations,
                     'expanded_mask': expanded_mask,
                     'encoder_attention': encoder_attention,
                     'decoder_attention': decoder_attention}
        return model_out
    
    def set_constants(self, learning_rate: float = None, **kwargs):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
    
    def encode_text(self, text):
        return self.text_pipeline(text)
    
    def predict(self, inp, spk_emb, encode=True, speed_regulator=1., phoneme_max_duration=None, phoneme_min_duration=None,
                max_durations_mask=None, min_durations_mask=None, phoneme_durations=None):
        if encode:
            inp = self.encode_text(inp)
        if len(tf.shape(inp)) < 2:
            inp = tf.expand_dims(inp, 0)
        inp = tf.cast(inp, tf.int32)
        duration_scalar = tf.cast(1. / speed_regulator, tf.float32)
        max_durations_mask = self._make_max_duration_mask(inp, phoneme_max_duration)
        min_durations_mask = self._make_min_duration_mask(inp, phoneme_min_duration)
        out = self.call(inp,
                        target_durations=phoneme_durations,
                        spk_emb=spk_emb,
                        training=False,
                        durations_scalar=duration_scalar,
                        max_durations_mask=max_durations_mask,
                        min_durations_mask=min_durations_mask)
        out['mel'] = tf.squeeze(out['mel'])
        return out
    
    def _make_max_duration_mask(self, encoded_text, phoneme_max_duration):
        np_text = np.array(encoded_text)
        new_mask = np.ones(tf.shape(encoded_text)) * float('inf')
        if phoneme_max_duration is not None:
            for item in phoneme_max_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)
    
    def _make_min_duration_mask(self, encoded_text, phoneme_min_duration):
        np_text = np.array(encoded_text)
        new_mask = np.zeros(tf.shape(encoded_text))
        if phoneme_min_duration is not None:
            for item in phoneme_min_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)
