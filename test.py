import os

from official.nlp import bert
from official.nlp.bert import tokenization

import constants

vocab_file = os.path.join(constants.LOCAL_FOLDER_BERT, "vocab.txt")

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=vocab_file,
    do_lower_case=True)

print(tokenizer.convert_ids_to_tokens([0]))
print(tokenizer("Hello, how are you today?"))


