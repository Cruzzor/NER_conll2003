import os
from typing import Tuple


class InputParser:
    """
    Reads in the data and parses the CSV, transforming the input
    into the desired output structure.
    """
    def __init__(self, input_data_path: str):
        self._input_data_path = input_data_path
        self._validate_file_path()

    def _validate_file_path(self) -> None:
        if not os.path.exists(self._input_data_path):
            raise FileNotFoundError(self._input_data_path)

    def parse_data(self) -> Tuple[list, list]:
        """
        Parses the raw csv file. The outputs are two lists:
        1.) sentences: Each sentence is split into individual words and therefore
            again a list of strings

        2.) ner_tags: Each sublist is the corresponding sequence of NER tags for
            the individual words of the corresponding sentence
        """
        unique_tags = []

        sentences = []
        ner_tags = []

        sentence = []
        tag_sequence = []

        for line in open(self._input_data_path, 'r'):
            line = line.strip()
            if not line.startswith('-DOCSTART-'):
                if line != '':
                    word, ner_tag = self._parse_input_line(line)
                    sentence.append(word)
                    tag_sequence.append(ner_tag)
                    if ner_tag not in unique_tags:
                        unique_tags.append(ner_tag)
                else:
                    if len(sentence) != 0:
                        sentences.append(sentence)
                        ner_tags.append(tag_sequence)
                    sentence = []
                    tag_sequence = []

        return sentences, ner_tags

    def _parse_input_line(self, line: str) -> Tuple[str, str]:
        parts = line.split(' ')
        word = parts[0]
        ner_tag = parts[3]

        return word, ner_tag
