import os
from dataclasses import dataclass
from glob import glob

import fire
from blingfire import text_to_sentences
from official.nlp import bert
from official.nlp.bert import tokenization

import constants


def to_sentence_lines(lines):
    paragraph = []
    document = []
    num_sents = 0

    for line in lines:
        line = line.strip()
        if not line:
            sents = text_to_sentences(" ".join(paragraph).strip().replace("\n", ''))
            sents = sents.split('\n')
            document.extend(sents)
            num_sents += len(sents)
            paragraph = []
            continue

        paragraph.append(line.strip())

    if paragraph:
        sents = text_to_sentences(" ".join(paragraph).strip().replace("\n", ''))
        sents = sents.split('\n')
        document.extend(sents)
        num_sents += len(sents)

    return [sen for sen in document if sen != ''], num_sents


def run(input_dir:str, output_dir:str):
    file_list = list(sorted(glob(os.path.join(input_dir, '*.txt'))))

    for file in file_list:
        file_name = os.path.basename(file)
        print(f'Preprocessing document {file_name}.')
        lines = open(file, encoding='utf-8').readlines()
        lines, num_lines = to_sentence_lines(lines)
        print(f'Converted to {num_lines} lines.')

        with open(os.path.join(output_dir, file_name), "w") as f:
            f.write("\n".join(lines))
            print(f"Finished writing to {os.path.join(output_dir, file_name)}")
        print('\n')


if __name__ == "__main__":
    fire.Fire(run)
