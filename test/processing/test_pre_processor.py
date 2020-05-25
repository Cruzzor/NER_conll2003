import numpy as np
import pytest

from numpy.testing import assert_array_equal
from src.processing.PreProcessor import PreProcessor


@pytest.fixture
def input_data():
    in_sentences = [
        ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE',
         'DEFEAT', '.'],
        ['Nadim', 'Ladki']
    ]

    in_tags = [
        ['O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'I-PER', 'O', 'O', 'O', 'O'],
        ['I-PER', 'I-PER']
    ]

    return in_sentences, in_tags


def test_pre_process_data_training(input_data):
    in_sentences, in_tags = input_data

    pre_processor = PreProcessor(in_sentences, in_tags, in_sentences, in_tags)
    pre_processor.pre_process_data()

    expected_input_sequences = np.array([
        [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
        [14, 15, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])

    expected_label_sequences = np.array([
        [1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    expected_num_unique_word_tokens = 16  # Unique words + one oov token + padding token

    expected_num_unique_label_tokens = 3

    assert_array_equal(pre_processor.input_sequences, expected_input_sequences)
    assert_array_equal(pre_processor.label_sequences, expected_label_sequences)

    assert pre_processor.num_unique_word_tokens == expected_num_unique_word_tokens
    assert pre_processor.num_unique_label_tokens == expected_num_unique_label_tokens


def test_pre_process_data_validation(input_data):
    in_sentences, in_tags = input_data

    in_val_sentences = [
        ['SOCCER', 'JAPAN', 'CHINA', 'UNKNOWN'],
        ['Nadim', 'unknown_name']
    ]

    in_val_tags = [
        ['O', 'I-LOC', 'I-PER', 'O'],
        ['I-PER', 'I-PER']
    ]

    pre_processor = PreProcessor(in_sentences, in_tags, in_val_sentences, in_val_tags)
    pre_processor.pre_process_data()

    expected_input_sequences = np.array([
        [2, 4, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    expected_label_sequences = np.array([
        [1, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    assert_array_equal(pre_processor.val_input_sequences, expected_input_sequences)
    assert_array_equal(pre_processor.val_label_sequences, expected_label_sequences)
