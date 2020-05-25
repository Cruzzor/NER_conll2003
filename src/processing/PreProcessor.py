import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PreProcessor:
    """
    Transforms the parsed text data into the correct format and shape
    to be used as input for the model.
    """
    def __init__(self, sentences, ner_tags, val_sentences, val_ner_tags, oov_token: str = "<OOV>"):
        self._sentences = sentences
        self._ner_tags = ner_tags
        self._val_sentences = val_sentences
        self._val_ner_tags = val_ner_tags

        self._input_sequences = None
        self._label_sequences = None
        self._val_input_sequences = None
        self._val_label_sequences = None

        self._tokenizer = Tokenizer(oov_token=oov_token)
        self._label_tokenizer = Tokenizer()

        self._max_sequence_length = None

    def pre_process_data(self):
        self._pre_process_train_input_sequences()
        self._pre_process_train_label_sequences()

        self._pre_process_validation_input_sequences()
        self._pre_process_validation_label_sequences()

    def _pre_process_train_input_sequences(self):
        self._input_sequences = self._pre_process_input_sequence(self._sentences)

    def _pre_process_train_label_sequences(self):
        self._label_sequences = self._pre_process_label_sequences(self._ner_tags)

    def _pre_process_validation_input_sequences(self):
        self._val_input_sequences = self._pre_process_input_sequence(self._val_sentences,
                                                                     validation=True)

    def _pre_process_validation_label_sequences(self):
        self._val_label_sequences = self._pre_process_label_sequences(self._val_ner_tags,
                                                                      validation=True)

    def _pre_process_input_sequence(self, sentences: list, validation=False) -> np.ndarray:
        if not validation:
            self._tokenizer.fit_on_texts(sentences)
            self._compute_max_sequence_len(sentences)
        input_sequences = self._tokenizer.texts_to_sequences(sentences)
        padded_input_sequences = pad_sequences(input_sequences, padding='post',
                                               maxlen=self._max_sequence_length)
        return np.array(padded_input_sequences)

    def _pre_process_label_sequences(self, ner_tags: list, validation: bool = False) -> np.ndarray:
        if not validation:
            self._label_tokenizer.fit_on_texts(ner_tags)
        label_sequences = self._label_tokenizer.texts_to_sequences(ner_tags)
        padded_label_sequences = pad_sequences(label_sequences, padding='post',
                                               maxlen=self._max_sequence_length)
        return np.array(padded_label_sequences)

    def _compute_max_sequence_len(self, input_sequences: list):
        seq_lengths = [len(seq) for seq in input_sequences]
        self._max_sequence_length = max(seq_lengths)

    @property
    def input_sequences(self):
        return self._input_sequences

    @property
    def label_sequences(self):
        return self._label_sequences

    @property
    def val_input_sequences(self):
        return self._val_input_sequences

    @property
    def val_label_sequences(self):
        return self._val_label_sequences

    @property
    def num_unique_word_tokens(self):
        return len(self._tokenizer.word_index) + 1  # +1 for padding token

    @property
    def num_unique_label_tokens(self):
        return len(self._label_tokenizer.word_index)

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def label_index_to_words_dict(self):
        return self._label_tokenizer.index_word
