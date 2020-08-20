import os

import tensorflow as tf

from official.nlp import bert
from official.nlp.bert import tokenization

import constants

vocab_file = os.path.join(constants.LOCAL_FOLDER_BERT, "vocab.txt")

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=vocab_file,
    do_lower_case=True)

print(tokenizer.convert_ids_to_tokens([0]))
print(tokenizer.tokenize("Hello, how are you today?"))

path = "C://Users//alexs//Documents//GitHub//multi-word-generator//documents//processed"
file = "data_chunk_0"


data = tf.data.TFRecordDataset(filenames=os.path.join(path, file))
print(data)

for raw_data in data.take(10):
  print(repr(raw_data))

  # Create a description of the features.
  feature_description = {
      'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
  }


  def _parse_function(example_proto):
      # Parse the input `tf.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, feature_description)


  parsed_dataset = raw_dataset.map(_parse_function)
  parsed_dataset
