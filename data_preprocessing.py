import os
import tensorflow as tf

from official.nlp import bert

import official.nlp.bert.tokenization

# local imports
import constants

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(constants.LOCAL_FOLDER_BERT, "vocab.txt"),
    do_lower_case=True)


class TrainingInstance:

    def __init__(self,
                 tokens,
                 masked_tokens=None):

        assert tokens.contains('[MASK]')
        # NOTE maybe it could also work without having the [MASK] input during runtime
        # As in letting the model decide where to generate new words
        self.tokens = tokens
        if masked_tokens is None:
            masked_tokens = []
        self.masked_tokens = masked_tokens
        self.num_masked_tokens = len(masked_tokens)

    def __str__(self):
        #TODO implement printable version

def create_instances(document_files,
                     tokenizer,
                     max_seq_length,
                     max_masked_length,
                     short_seq_prob,
                     rng,):
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

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for i, doc in all_documents:
        instances.extend(
            create_instances_from_document(
                all_documents, i, max_seq_length,
                short_seq_prob, vocab_words, rng))

        rng.shuffle(instances)
        return instances


def create_instances_from_document(documents,
                                   document_index,
                                   max_seq_length,
                                   short_seq_prob,
                                   vocab_words,
                                   rng):
    pass


def tokenize_sentence(sen):
    tokens = list(tokenizer.tokenize(sen.numpy()))
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs




# TODO Lookup the BERT training Data generation
# TODO make similar junks where between zero and 5 words are hidden by a [MASK] token
# TODO create expected vector with reverse ranks at the id of the word, multi-hot-encoding

# CLS can be used for a regression to how many words should be introduced
tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

input_mask = tf.ones_like(input_word_ids).to_tensor()

# TODO I do not need the seperator at the end, since I am only dealing with one sentence
# TODO The input type in my case will be all zeros

type_cls = tf.zeros_like(cls)
type_s1 = tf.zeros_like(sentence1)
type_s2 = tf.ones_like(sentence2)
input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

