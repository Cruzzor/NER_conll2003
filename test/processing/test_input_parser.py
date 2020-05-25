import pytest

from src.processing.InputParser import InputParser


def test_parse_data():
    in_test_data_path = "./test_input.txt"

    input_parser = InputParser(in_test_data_path)
    sentences, ner_tags = input_parser.parse_data()

    expected_sentences = [
        ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE',
         'DEFEAT', '.'],
        ['Nadim', 'Ladki']
    ]

    expected_ner_tags = [
        ['O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'I-PER', 'O', 'O', 'O', 'O'],
        ['I-PER', 'I-PER']
    ]

    assert sentences == expected_sentences
    assert ner_tags == expected_ner_tags


def test_parse_data_file_not_existing():
    in_test_data_path = "NO_VALID_FILE_PATH"

    with pytest.raises(FileNotFoundError):
        InputParser(in_test_data_path)
