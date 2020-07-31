import os
import math
import logging
import random
from dataclasses import dataclass, field
from typing import List
from itertools import chain
from collections import OrderedDict

import fire
import numpy as np
import tensorflow as tf
from official.nlp import bert
import official.nlp.bert.tokenization

# local imports
import constants


@dataclass
class TrainingInstance:
    """Used for saving and loading from a file

    Distance_ids mark how far the words are away from the [MASK] token and can be used for embeddings.
    """
    tokens: List[str]
    distance_ids: List[int]
    mask_token_pos: int
    masked_tokens: List[int] = None
    num_masked_tokens: int = field(init=False)

    def __post_init__(self):
        assert '[MASK]' in self.tokens

        if self.masked_tokens is None:
            self.masked_tokens = []
        self.num_masked_tokens = len(self.masked_tokens)

    def __str__(self):
        tokens = self.tokens
        return " ".join(tokens)[:-1]


def create_training_instances(document_files,
                              tokenizer,
                              max_seq_length,
                              max_masked_tokens,
                              dupe_factor,
                              rng):
    """
    Create training instances from multiple documents.
    The data format is (1) Each document is in it's own file.
    (2) Each sentence is in it's own line.
    """
    all_documents = []

    for file in document_files:
        with open(file, "r") as reader:
            all_documents.append([])
            for line in reader.readlines():
                line = line.strip()

                if not line:
                    continue

                tokens = tokenizer(line)
                all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for _ in range(dupe_factor):
        for i, doc in all_documents:
            instances.extend(
                create_instances_from_document(
                    all_documents[i], tokenizer,
                    max_seq_length, max_masked_tokens, rng))

    rng.shuffle(instances)
    return instances


def random_num_masked_tokens(rng, max_masked_tokens):
    if rng.random() < 0.05:
        return 0
    else:
        return rng.randint(1, max_masked_tokens)


def create_instances_from_document(document,
                                   tokenizer,
                                   max_seq_length,
                                   max_masked_tokens,
                                   rng,
                                   long_seq_prob=0.95,
                                   geo_dist_p_value=0.1):
    """Creates `TrainingInstance`s for a single document."""
    num_sentences = len(document)
    document = list(chain(document))
    num_tokens = len(document)

    # num_masked_tokens is ground-truth for the [CLS] output to predict
    num_masked_tokens = random_num_masked_tokens(rng, max_masked_tokens)

    # The mask token will be put in max_seq_length // 2, which is the middle
    # because the leftmost slot will have the [CLS] token.

    # [MASK] replaces num_masked_tokens
    max_num_tokens = max_seq_length - num_masked_tokens + 1

    # To make the model more robust to changing sequence lengths,
    # the context to the left and right will vary in length.
    # Words will be substracted on both sides. In long_seq_prob amount
    # of times this number will be drawn from a geometric distribution.
    # For the rest of the the length will be uniformly distributed between
    # 2 and 0.1 * max_seq_length
    max_left_right_context = (max_seq_length - 1) // 2
    if rng.random() < long_seq_prob:
        sub_words = np.random.geometric(geo_dist_p_value)
        num_left_right_context = max_left_right_context - sub_words
    else:
        num_left_right_context = rng.randint(2, 0.1 * (max_seq_length - 2))

    i = num_left_right_context + math.ceil(num_masked_tokens / 2)
    max_index = len(document) - num_left_right_context - math.ceil(num_masked_tokens / 2)
    create_inst_prob = 3 * num_sentences / num_tokens
    instances = []

    while i < max_index:
        if rng.random() < create_inst_prob:
            tokens = [""] * max_seq_length
            token_ids = [0] * max_seq_length
            distance_ids = [0] * max_seq_length

            masked_token_shift = num_masked_tokens // 2

            # TODO this needs to be fixed for subword models, so that only whole words are masked
            if num_masked_tokens == 0:
                left_context = document[i - num_left_right_context: i]
                right_context = document[i: i + num_left_right_context]
                masked_tokens = []
            else:
                is_even = num_masked_tokens % 2
                if is_even:
                    left_context = document[i - num_left_right_context - masked_token_shift: i - masked_token_shift]
                    right_context = document[i + masked_token_shift: i + masked_token_shift + num_left_right_context]
                    masked_tokens = document[i - masked_token_shift: i + masked_token_shift]
                else:
                    # in this case take the odd token is taken from the left side
                    left_context = document[
                                   i - num_left_right_context - masked_token_shift - 1: i - masked_token_shift - 1]
                    right_context = document[i + masked_token_shift: i + masked_token_shift + num_left_right_context]
                    masked_tokens = document[i - masked_token_shift - 1: i + masked_token_shift]

            tokens[0] = "[CLS]"
            token_ids[0] = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]

            tokens[max_left_right_context + 1] = "[MASK]"
            token_ids[0] = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]

            dist_ascending = range(1, num_left_right_context + 1)

            start_index = max_left_right_context + 1 - num_left_right_context
            end_index = max_left_right_context + 1
            tokens[start_index: end_index] = left_context
            token_ids[start_index: end_index] = tokenizer.convert_tokens_to_ids(left_context)
            distance_ids[start_index: end_index] = reversed(dist_ascending)

            start_index = max_left_right_context + 2
            end_index = max_left_right_context + 2 + num_left_right_context
            tokens[start_index: end_index] = right_context
            token_ids[start_index: end_index] = tokenizer.convert_tokens_to_ids(right_context)
            distance_ids[start_index: end_index] = list(dist_ascending)

            instance = TrainingInstance(
                tokens=tokens,
                distance_ids=distance_ids,
                mask_token_pos=max_seq_length // 2,
                masked_tokens=masked_tokens)
            instances.append(instance)
        i += 1

    return instances


def create_ordered_vocab_vec(masked_tokens, tokenizer):
    res = [0] * tokenizer.vocab
    pos = len(masked_tokens)
    for token in masked_tokens:
        res[token] = pos
        pos -= 1
    return res


def write_instance_to_example_files(instances,
                                    tokenizer,
                                    max_seq_length,
                                    output_files):
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        distance_ids = instance.distance_ids
        assert len(input_ids) == max_seq_length
        assert len(distance_ids) == max_seq_length

        num_masked_tokens = instance.num_masked_tokens

        masked_tokens = instance.masked_tokens
        masked_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)
        masked_tokens = create_ordered_vocab_vec(masked_tokens, tokenizer)

        features = OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["distance_ids"] = create_int_feature(distance_ids)
        features["num_masked_tokens"] = create_int_feature(num_masked_tokens)
        features["masked_tokens"] = create_int_feature(masked_tokens)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            logging.info("*** Example ***")
            logging.info("tokens: %s" % " ".join(
                [tokenizer.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


@dataclass
class App:
    """Create training instances for mulit-word-generator.

    :Arguments
        input_path: Path to the data folder that keeps the text in individual text files.
        output_path: Path where to save the training instances.
        random_seed: Random seed for the random number generator.
        vocab_file: Path to vocab file.
        max_seq_length: Maximal number of input sequence length.
        max_masked_tokens: Maximal amount of tokens masked.
        dupe_factor: Duplication factor of each document when generating training instances.
    """
    input_path: str
    output_path: str #TODO make this into output folder and auto-generate 300MB sized files (if necesarry)
    random_seed: int = 1234
    vocab_file: str = os.path.join(constants.LOCAL_FOLDER_BERT, "vocab.txt")
    max_seq_length: int = 256
    max_masked_tokens: int = 5
    dupe_factor: int = 5

    def run(self):
        logging.info("App fired.")
        tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=self.vocab_file,
            do_lower_case=True)

        input_files = os.listdir(self.input_path)

        logging.info("### Reading files from input ###")
        for input_file in input_files:
            logging.info(f'    {input_file}')

        rng = random.Random(self.random_seed)

        instances = create_training_instances(
            input_files, tokenizer, self.max_seq_length,
            self.max_masked_tokens, self.dupe_factor,
            rng)

        logging.info("### Writing to output files ###")
        logging.info(f'    {self.output_path}')

        write_instance_to_example_files(
            instances, tokenizer, self.max_seq_length, [self.output_path])


if __name__ == "__main__":
    fire.Fire(App)
