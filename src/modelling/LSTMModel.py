import numpy as np

from sklearn.metrics import classification_report
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Sequential


class LSTMModel:
    """
    A simple LSTM model.
    Input:
        X: Tokenized, padded sentences
        Y: the corresponding NER token sequence
    """
    def __init__(self,
                 train_input_sequences: np.ndarray,
                 train_label_sequences: np.ndarray,
                 validation_input_sequences: np.ndarray,
                 validation_label_sequences: np.ndarray,
                 num_unique_word_tokens: int,
                 num_unique_label_tokens: int,
                 max_sequence_length: int,
                 label_index_to_word_map: dict):
        self._train_input_sequences = train_input_sequences
        self._train_label_sequences = train_label_sequences

        self._val_input_sequences = validation_input_sequences
        self._val_label_sequences = validation_label_sequences

        self._label_index_to_word_map = label_index_to_word_map
        self._model = None

        self._compile_model(num_unique_word_tokens, num_unique_label_tokens, max_sequence_length)

    def _compile_model(self,
                       num_unique_word_tokens: int,
                       num_unique_label_tokens: int,
                       max_sequence_length: int):
        embedding_size = 300

        model_layers = [
            Embedding(input_dim=num_unique_word_tokens, output_dim=embedding_size,
                      input_length=max_sequence_length, mask_zero=True),
            Bidirectional(LSTM(150, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)),
            TimeDistributed(Dense(num_unique_label_tokens + 1, activation='softmax'))
        ]

        self._model = Sequential(model_layers)
        self._model.compile(loss='sparse_categorical_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

    def train(self):
        history = self._model.fit(self._train_input_sequences, self._train_label_sequences,
                                  batch_size=50, epochs=1, verbose=1)
        print(history.history)

    def validate(self):
        model_out = self._model.predict(self._val_input_sequences)
        predictions = model_out.argmax(axis=-1)

        report = self._create_classification_report(predictions)

        print(report)

    def _create_classification_report(self, predictions: np.ndarray) -> str:
        y_true = self._val_label_sequences.flatten()
        y_hat = predictions.flatten()

        y_true_mapped = self._map_label_indices_to_words(y_true)
        y_hat_mapped = self._map_label_indices_to_words(y_hat)

        return classification_report(y_true_mapped, y_hat_mapped)

    def _map_label_indices_to_words(self, y: np.ndarray) -> np.ndarray:
        return np.vectorize(self._label_index_to_word_map.get)(y)
