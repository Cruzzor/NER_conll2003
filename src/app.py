from src.processing.InputParser import InputParser
from src.processing.PreProcessor import PreProcessor
from src.modelling.LSTMModel import LSTMModel


def train_model(training_data: str, validation_data: str):
    train_parser = InputParser(training_data)
    sentences, ner_tags = train_parser.parse_data()

    validation_parser = InputParser(validation_data)
    val_sentences, val_ner_tags = validation_parser.parse_data()

    pre_processor = PreProcessor(sentences, ner_tags, val_sentences, val_ner_tags)
    pre_processor.pre_process_data()

    lstm_model = LSTMModel(
        pre_processor.input_sequences,
        pre_processor.label_sequences,
        pre_processor.val_input_sequences,
        pre_processor.val_label_sequences,
        pre_processor.num_unique_word_tokens,
        pre_processor.num_unique_label_tokens,
        pre_processor.max_sequence_length,
        pre_processor.label_index_to_words_dict
    )

    lstm_model.train()

    return lstm_model


if __name__ == "__main__":
    DATA = "../../conll2003/eng.train"
    VAL_DATA = "../../conll2003/eng.testa"
    model = train_model(DATA, VAL_DATA)
    classification_report = model.validate()
