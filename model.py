import os
import json

import tensorflow as tf

from official.nlp import bert

import official.nlp.bert.configs

# local imports
import constants

bert_config_file = os.path.join(constants.LOCAL_FOLDER_BERT_ASSETS, "bert_config.json")
with open(bert_config_file, "r+") as f:
    config_dict = json.loads(f.read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

print(config_dict)