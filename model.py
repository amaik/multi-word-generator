import json
import os

import fire
import official.nlp.bert.configs
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import bert

import constants

# show download progressbar
TFHUB_PROGRESS_VAR = 'TFHUB_DOWNLOAD_PROGRESS'
os.environ[TFHUB_PROGRESS_VAR] = "1"


def bert_config_from_file(bert_config_file):
    config_dict = json.loads(open(bert_config_file, "r+").read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    return bert_config


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def multi_word_model(bert_config,
                     max_seq_length,
                     hub_url_bert_encoder,
                     hub_module_trainable,
                     final_layer_initializer=None):
    if final_layer_initializer is not None:
        initializer = final_layer_initializer
    else:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range)

    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    bert_model = hub.KerasLayer(hub_url_bert_encoder, trainable=hub_module_trainable)
    pooled_out, seq_out = bert_model([input_word_ids, input_mask, input_type_ids])

    pooled_out = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(pooled_out)
    pooled_out = tf.keras.layers.Dense(bert_config.hidden_size, activation=gelu)(pooled_out)
    pooled_out = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="output_num_words")(
        pooled_out)  # indicate the number of words to generate

    seq_out = tf.keras.layers.Flatten()(seq_out)
    seq_out = tf.keras.layers.Dropout(bert_config.hidden_dropout_prob)(seq_out)
    seq_out = tf.keras.layers.Dense(bert_config.hidden_size, kernel_initializer=initializer)(
        seq_out)  # TODO smaller intermediate layer for less weights, how much can and should I increase this?
    seq_out = tf.keras.layers.Dense(bert_config.vocab_size, kernel_initializer=initializer, name="output_order_words")(
        seq_out)

    return tf.keras.Model(
        inputs={
            'input_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs={
            'output_num_gapped': pooled_out,
            'output_order': seq_out
        })


def plot_model(bert_config_file=os.path.join(constants.LOCAL_FOLDER_BERT, "bert_config.json")):
    bert_config = bert_config_from_file(bert_config_file)

    model = multi_word_model(bert_config)
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=48)


if __name__ == "__main__":
    fire.Fire(plot_model)
