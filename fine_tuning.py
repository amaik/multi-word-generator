import os

import fire
import tensorflow as tf

import constants
from model import multi_word_model, bert_config_from_file


def load_record_data(input_dir,
                     max_seq_length,
                     vocab_size,
                     batch_size):
    input_files = []
    for file in os.listdir(input_dir):
        input_files.append(os.path.join(input_dir, file))

    record_data = tf.data.TFRecordDataset(filenames=input_files)

    def _parse_examples(example):
        feature_description = {
            'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'input_type_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'input_distance_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'output_num_gapped': tf.io.FixedLenFeature([1], tf.int64),
            'output_order': tf.io.FixedLenFeature([vocab_size], tf.int64)
        }

        parsed_example = tf.io.parse_single_example(example, feature_description)

        # input_ids = parsed_example['input_ids']
        # distance_ids = parsed_example['distance_ids']
        # num_gapped_tokens = parsed_example['num_gapped_tokens']
        # gapped_tokens = parsed_example['gapped_tokens']
        #
        # return input_ids, distance_ids, num_gapped_tokens, gapped_tokens
        return parsed_example

    # TODO try this again with just tf.io.parse_example
    # feature_tensors = tf.io.parse_example(example_batch, feature_specification)

    parsed_data = record_data.map(_parse_examples)
    parsed_data.batch(constants.TRAIN_BATCH_SIZE)
    print(dir(parsed_data.take(1)))
    print(parsed_data.element_spec)

    return parsed_data


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
    :param train_batch_size:
    :param hub_url_bert_encoder:
    :param hub_module_trainable:
    :param final_layer_initializer:
    :return:
    """
    bert_config = bert_config_from_file(bert_config_file)

    data_set = load_record_data(input_dir,
                                max_seq_length,
                                bert_config.vocab_size,
                                train_batch_size)

    model = multi_word_model(bert_config,
                             max_seq_length,
                             hub_url_bert_encoder,
                             hub_module_trainable,
                             final_layer_initializer)

    model.compile(optimizer=tf.keras.optimizers.Adam,
                  loss={
                      'output_num_words': tf.keras.losses.mse,
                      'output_order_words': tf.keras.losses.mse
                  },
                  metrics={
                      'output_num_words': tf.keras.metrics.mae,
                      'output_order_words': tf.keras.metrics.mae
                  })

    # TODO fit the model
    model.fit_generator()


if __name__ == "__main__":
    fire.Fire(load_record_data)

    # TODO use tensorboard when training this

    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create a new model to get a fresh trace
    # Otherwise the summary will not see the graph.
    new_model = MySequentialModule()

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])))
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)

    % tensorboard - -logdir logs/func
