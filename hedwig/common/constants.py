import os

# Model categories
BERT_MODELS = ['BERT-Base', 'BERT-Large', 'HBERT-Base', 'HBERT-Large']

# String templates for logging results
LOG_HEADER = 'Split  Dev/Loss   Dev/mse'
LOG_TEMPLATE = ' '.join('{:>5s},,{:10.4f},{:10.4f}'.split(','))

# Path to pretrained model and vocab files
MODEL_DATA_DIR = os.path.join(os.pardir, 'hedwig-data', 'models')
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased')
}
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased-vocab.txt'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased-vocab.txt'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased-vocab.txt'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased-vocab.txt'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased-vocab.txt'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased-vocab.txt')
}
