import os


GS_FOLDER_BERT = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
LOCAL_FOLDER_BERT = "C:/Users/alexs/models/bert/uncased_L-12_H-768_A-12"
HUB_URL_BERT = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


EPOCHS = 3
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

BERT_CONFIG = r'''{
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522
}'''