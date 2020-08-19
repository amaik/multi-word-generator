import os

import fire
import tensorflow as tf

import constants
from model import multi_word_model, bert_config_from_file


def input_generator():
    """
    TODO Write an input generator
    :return:
    """


def train(input_dir,
          output_dir,
          bert_config_file=os.path.join(constants.LOCAL_FOLDER_BERT, "bert_config.json"),
          max_seq_length=constants.MAX_SEQ_LENGTH,
          train_batch_size=constants.TRAIN_BATCH_SIZE,
          hub_url_bert_encoder=constants.HUB_URL_BERT,
          hub_module_trainable=True,
          final_layer_initializer=None):
    """

    :param input_dir:
    :param output_dir:
    :param bert_config_file:
    :param max_seq_length:
    :param hub_url_bert_encoder:
    :param hub_module_trainable:
    :param final_layer_initializer:
    :return:
    """
    bert_config = bert_config_from_file(bert_config_file)
    model = multi_word_model(bert_config,
                             max_seq_length,
                             hub_url_bert_encoder,
                             hub_module_trainable,
                             final_layer_initializer)


    model.compile(optimizer=tf.keras.optimizers.Adam,
                  loss={
                      'output_num_words': tf.keras.losses.mse,
                      'output_order_words': tf.keras.losses.mse
                  })


if __name__ == "__main__":
    fire.Fire(train)

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / constants.BATCH_SIZE)
num_train_steps = steps_per_epoch * constants.EPOCHS
warmup_steps = int(constants.EPOCHS * train_data_size * 0.1 / constants.BATCH_SIZE)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
    glue_train, glue_train_labels,
    validation_data=(glue_validation, glue_validation_labels),
    batch_size=32,
    epochs=epochs)
