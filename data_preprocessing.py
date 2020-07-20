import os
import tensorflow as tf

from official.nlp import bert

import official.nlp.bert.tokenization



tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join('C:/Users/alexs/models/bert/uncased_L-12_H-768_A-12_2/assets', "vocab.txt"),
     do_lower_case=True)

print(tokenizer.convert_tokens_to_ids(['[SEP]']))