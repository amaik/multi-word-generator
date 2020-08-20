import cProfile
import logging
import math
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import List

import fire
import numpy as np
import tensorflow as tf
from official.nlp import bert
from official.nlp.bert import tokenization
from tqdm import tqdm

# local imports
import constants

DEBUG = None
PROFILING = None


@dataclass
class TrainingInstance:
    """Used for saving and loading from a file

    Distance_ids mark how far the words are away from the [GAP] token and can be used for embeddings.
    Max_seq_length should be an even number.
    [CLS] will be the first token and [GAP] will be in position
    (max_seq_length - 1) // 2
    """
    tokens: List[str]
    distance_ids: List[int]
    gapped_tokens: List[str] = None
    num_gapped_tokens: int = field(init=False)

    def __post_init__(self):
        assert '[GAP]' in self.tokens

        if self.gapped_tokens is None:
            self.gapped_tokens = []
        self.num_gapped_tokens = len(self.gapped_tokens)

    def __str__(self):
        str = " ".join(self.tokens)
        str += f"\nGapped tokens = {self.gapped_tokens}"
        return str

    def token_ids(self, tokenizer):
        return tokenizer.convert_tokens_to_ids(self.tokens)

    def pretty_print(self, tokenizer):
        print(str(self), end="")
        print(f"Token Ids = {self.token_ids(tokenizer)}")
        print(f"Distance Ids = {self.distance_ids}")


def create_training_instances(input_dir,
                              tokenizer,
                              max_seq_length,
                              max_gapped_tokens,
                              dupe_factor,
                              rng):
    """
    Create training instances from multiple documents.
    The data format is (1) Each document is in it's own file.
    (2) Each sentence is in it's own line.
    """
    all_documents = []

    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), "r+") as reader:
            all_documents.append([])
            for line in reader.readlines():
                line = tokenization.convert_to_unicode(line)
                line = line.strip()

                if not line:
                    continue

                tokens = tokenizer.tokenize(line)
                all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for _ in range(dupe_factor):
        for i, doc in enumerate(all_documents):
            instances.extend(
                create_instances_from_document(
                    all_documents[i], max_seq_length,
                    max_gapped_tokens, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(document,
                                   max_seq_length,
                                   max_gapped_tokens,
                                   rng,
                                   long_seq_prob=0.95,
                                   geo_dist_p_value=0.1):
    """Creates `TrainingInstance`s for a single document.

        The [CLS] output will predict len(gapped_tokens). The gap
        token will be put in max_seq_length // 2, which is the middle
        because the leftmost slot will have the [CLS] token. [GAP] replaces
        the gapped tokens in the output.
        To make the model more robust to changing sequence lengths,
        the context to the left and right will vary in length.
        Words will be substracted on both sides. In long_seq_prob amount
        of times this number will be drawn from a geometric distribution.
        For the rest of the the length will be uniformly distributed between
        2 and 0.1 * max_seq_length.
    """

    def _random_num_gapped_tokens():
        nonlocal rng
        nonlocal max_gapped_tokens
        if rng.random() < 0.05:
            return 0
        else:
            return rng.randint(1, max_gapped_tokens)

    logging.debug("Started creating instances for document.")

    num_sentences = len(document)
    document = list(chain(*document))
    num_tokens = len(document)

    max_left_right_context = (max_seq_length - 1) // 2
    i = max_left_right_context + math.ceil(max_gapped_tokens / 2)
    max_index = len(document) - max_left_right_context - math.ceil(max_gapped_tokens / 2)
    create_inst_prob = num_sentences / num_tokens  # creates approximately num_sentences instances per duplication
    instances = []

    while i < max_index:
        if rng.random() < create_inst_prob:
            logging.debug(f"Creating training instance at index {i}.")

            tokens = ['[PAD]'] * max_seq_length
            distance_ids = [0] * max_seq_length

            if rng.random() < long_seq_prob:
                sub_words = np.random.geometric(geo_dist_p_value)
                num_left_right_context = max_left_right_context - sub_words
            else:
                num_left_right_context = rng.randint(2, (max_seq_length - 2) // 10)

            desired_num_gapped_tokens = _random_num_gapped_tokens()
            gapped_tokens, left_context, right_context = gap_tokens(document,
                                                                    desired_num_gapped_tokens,
                                                                    i,
                                                                    num_left_right_context)
            logging.debug("Finished gapping tokens.")
            tokens[0] = "[CLS]"

            gap_pos = max_left_right_context + 1
            assert gap_pos == 128
            tokens[gap_pos] = "[GAP]"

            dist_ascending = range(1, num_left_right_context + 1)
            len_left_context = len(left_context)
            len_right_context = len(right_context)

            tokens[gap_pos - len_left_context: gap_pos] = left_context
            distance_ids[gap_pos - len_left_context: gap_pos] = list(reversed(dist_ascending))[-len_left_context:]

            post_gap = gap_pos + 1
            tokens[post_gap: post_gap + len_right_context] = right_context
            distance_ids[post_gap: post_gap + len_right_context] = list(dist_ascending)[0: len_right_context]

            assert distance_ids[gap_pos] == 0
            assert distance_ids[gap_pos + 1] == 1
            assert distance_ids[gap_pos - 1] == 1

            instance = TrainingInstance(
                tokens=tokens,
                distance_ids=distance_ids,
                gapped_tokens=gapped_tokens)

            instances.append(instance)
            logging.debug("Finished creating Instance.")
        i += 1

    return instances


def gap_tokens(document, num_gapped_tokens, i, num_left_right_context):
    def _shift_to_full_word():
        nonlocal document
        nonlocal start_index
        nonlocal end_index
        while True:
            if document[start_index].startswith('##'):
                start_index -= 1
            elif document[end_index].startswith('##'):
                end_index += 1
            else:
                return start_index, end_index

    logging.debug("Gapping tokens.")

    if num_gapped_tokens == 0:
        left_context = document[i - num_left_right_context: i]
        right_context = document[i: i + num_left_right_context]
        gapped_tokens = []
    else:
        is_even = num_gapped_tokens % 2
        if is_even:
            start_index = i - (num_gapped_tokens // 2)
        else:
            # in this case take the odd token is taken from the left side
            start_index = i - (num_gapped_tokens // 2) - 1

        end_index = i + (num_gapped_tokens // 2)
        logging.debug("Starting shift.")
        new_start_index, new_end_index = _shift_to_full_word()
        logging.debug("Shift finished.")

        gapped_tokens = document[new_start_index: new_end_index]
        left_context = document[new_start_index - num_left_right_context: new_start_index]
        right_context = document[new_end_index: new_end_index + num_left_right_context]

    return gapped_tokens, left_context, right_context


def write_instances(instances,
                    tokenizer,
                    max_seq_length,
                    output_dir):
    writers = []
    output_files = [f"data_chunk_{chunk}" for chunk in range(math.ceil(len(instances) / 10000))]
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(os.path.join(output_dir, output_file)))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(tqdm(instances)):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        distance_ids = instance.distance_ids
        assert len(input_ids) == max_seq_length
        assert len(distance_ids) == max_seq_length

        num_gapped_tokens = instance.num_gapped_tokens

        def _create_ordered_vocab_vec():
            nonlocal gapped_tokens
            nonlocal tokenizer
            res = [0] * len(tokenizer.vocab)
            pos = len(gapped_tokens)
            for token in gapped_tokens:
                res[token] = pos
                pos -= 1
            return res

        gapped_tokens = tokenizer.convert_tokens_to_ids(instance.gapped_tokens)
        gapped_tokens = _create_ordered_vocab_vec()

        input_mask = np.zeros_like(input_ids)
        input_mask[np.where(np.array(input_ids) != 0)] = 1
        input_type_ids = np.zeros_like(input_ids)

        def _create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        # TODO describe the variables
        features = OrderedDict()
        features["input_ids"] = _create_int_feature(input_ids)
        features["input_mask"] = _create_int_feature(input_mask)
        features["input_type_ids"] = _create_int_feature(input_type_ids)
        features["input_distance_ids"] = _create_int_feature(distance_ids)
        features["output_num_gapped"] = _create_int_feature([num_gapped_tokens])
        features["output_order"] = _create_int_feature(gapped_tokens)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1


        if DEBUG and inst_index < 20:
            logging.debug(f"*** Writing Example {inst_index}***")
            logging.info("tokens: %s" % " ".join(instance.tokens))

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


def run(input_dir: str,
        output_dir: str,
        random_seed: int = 1234,
        vocab_file: str = os.path.join(constants.LOCAL_FOLDER_BERT, "vocab.txt"),
        max_seq_length: int = constants.MAX_SEQ_LENGTH,
        max_gapped_tokens: int = 5,
        dupe_factor: int = 2,
        debug: bool = False,
        profiling: bool = False):
    """Create training instances for multi-word-generator.

    :Arguments
        input_path: Path to the data folder that keeps the text in individual text files.
        output_path: Path where to save the training instances.
        random_seed: Random seed for the random number generator.
        vocab_file: Path to vocab file.
        max_seq_length: Maximal number of input sequence length.
        max_gapped_tokens: Desired amount of maximal tokens in a gap. In some fringe cases the effective
                            amount of tokens in a gap might be slightly higher, due to word completion inside the gap.
        dupe_factor: Duplication factor of each document when generating training instances.
    """
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    global DEBUG
    global PROFILING
    DEBUG = debug
    PROFILING = profiling

    pr = None
    if PROFILING:
        pr = cProfile.Profile()
        pr.enable()

    logging.info("Application fired.")
    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=True)

    input_files = os.listdir(input_dir)

    logging.info("### Reading files from input ###")
    for input_file in input_files:
        logging.info(f'    {input_file}')

    rng = random.Random(random_seed)

    instances = create_training_instances(
        input_dir, tokenizer, max_seq_length,
        max_gapped_tokens, dupe_factor,
        rng)

    logging.info("### Writing to output files ###")
    logging.info(f'    {output_dir}')

    write_instances(instances, tokenizer, max_seq_length, output_dir)
    logger.info("### Finished writing output to files ###")

    if PROFILING:
        pr.disable()
        pr.print_stats(sort="calls")


if __name__ == "__main__":
    fire.Fire(run)
