# Import 3rd-party frameworks.

import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import time as tm
from datetime import datetime, timedelta
from matplotlib import pyplot as plt



def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation="selu", kernel_initializer="lecun_normal"),
      tf.keras.layers.Dense(d_model, activation="selu", kernel_initializer="lecun_normal"),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class Representation(tf.keras.layers.Layer):

  def __init__(self, *, complexity, d_model, dff, dropout_rate):
    super().__init__()
    self.complexity = complexity

    self.layers = [ FeedForward(d_model, dff, dropout_rate) 
           for _ in range(self.complexity) ]

  def call(self, x):
    # x = tf.keras.layers.LayerNormalization(x)

    for i in range(len(self.layers)):
      x = self.layers[i](x)

    return x  
  
class ConPositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, *, complexity, d_model, dff, dropout_rate):
    super().__init__()
    self.d_model = d_model

    self.representation = Representation(
      complexity=complexity, d_model=self.d_model, dff=dff, dropout_rate=dropout_rate)

    self.pos_encoding = positional_encoding(length=2048, depth=self.d_model)

  # def compute_mask(self, *args, **kwargs):
  #   return self.embedding.compute_mask(*args, **kwargs)


  def call(self, x):
    length = tf.shape(x)[1]
    x = self.representation(x)
    # This factor sets the relative scale of the representation and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
   

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
  
class ConEncoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, repComplexity, dropout_rate=0.1):
    super(ConEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = ConPositionalEmbedding(
      complexity=repComplexity,
      d_model=d_model,
      dff=dff,
      dropout_rate=dropout_rate
    )

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x 

class ConDecoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               repComplexity, dropout_rate=0.1):
    super(ConDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = ConPositionalEmbedding(
      complexity=repComplexity,
      d_model=d_model,
      dff=dff,
      dropout_rate=dropout_rate
    )
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)   
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x 
class ConTransformer(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               repComplexity, dropout_rate=0.1, name='trans'):
    super().__init__(name=name)
    self.encoder = ConEncoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff, repComplexity=repComplexity,
                           dropout_rate=dropout_rate)

    self.decoder = ConDecoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff, repComplexity=repComplexity,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(d_model)


  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs
    
    # content = tf.keras.layers.LayerNormalization()(context)
    # x = tf.keras.layers.LayerNormalization()(x)

    context = self.encoder(context)  # (batch_size, context_len, d_model) #-------------------- TEMP
    x = self.decoder(x, context)  # (batch_size, target_len, d_model) #----------------------- TEMP

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, d_model)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
  

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  


# class Translator(tf.Module):
#   def __init__(self, tokenizers, transformer):
#     self.tokenizers = tokenizers
#     self.transformer = transformer

#   def __call__(self, sentence, max_length=MAX_TOKENS):
#     # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
#     assert isinstance(sentence, tf.Tensor)
#     if len(sentence.shape) == 0:
#       sentence = sentence[tf.newaxis]

#     sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

#     encoder_input = sentence

#     # As the output language is English, initialize the output with the
#     # English `[START]` token.
#     start_end = self.tokenizers.en.tokenize([''])[0]
#     start = start_end[0][tf.newaxis]
#     end = start_end[1][tf.newaxis]

#     # `tf.TensorArray` is required here (instead of a Python list), so that the
#     # dynamic-loop can be traced by `tf.function`.
#     output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
#     output_array = output_array.write(0, start)

#     for i in tf.range(max_length):
#       output = tf.transpose(output_array.stack())
#       predictions = self.transformer([encoder_input, output], training=False)

#       # Select the last token from the `seq_len` dimension.
#       predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

#       predicted_id = tf.argmax(predictions, axis=-1)

#       # Concatenate the `predicted_id` to the output which is given to the
#       # decoder as its input.
#       output_array = output_array.write(i+1, predicted_id[0])

#       if predicted_id == end:
#         break

#     output = tf.transpose(output_array.stack())
#     # The output shape is `(1, tokens)`.
#     text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

#     tokens = tokenizers.en.lookup(output)[0]

#     # `tf.function` prevents us from using the attention_weights that were
#     # calculated on the last iteration of the loop.
#     # So, recalculate them outside the loop.
#     self.transformer([encoder_input, output[:,:-1]], training=False)
#     attention_weights = self.transformer.decoder.last_attn_scores

#     return text, tokens, attention_weights
  

#   translator = Translator(tokenizers, transformer)

  