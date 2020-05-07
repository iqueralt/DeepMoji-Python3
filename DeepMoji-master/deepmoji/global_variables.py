""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend(['{}BLANK_{}'.format(SPECIAL_PREFIX, i) for i in range(6, 10)])

ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/model/deepmoji_weights.hdf5'.format(ROOT_PATH)

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']

EMOJI_ENCODING = [
"\U0001f602",
"\U0001f612",
"\U0001f629",
"\U0001f62d",
"\U0001f60d",
"\U0001f614",
"\U0001f44c",
"\U0001f60a",
"\u2764",
"\U0001f60f",
"\U0001f601",
"\U0001f3b6",
"\U0001f633",
"\U0001f4af",
"\U0001f634",
"\U0001f60c",
"\u263a",
"\U0001f64c",
"\U0001f495",
"\U0001f611",
"\U0001f605",
"\U0001f64f",
"\U0001f615",
"\U0001f618",
"\u2665",
"\U0001f610",
"\U0001f481",
"\U0001f61e",
"\U0001f648",
"\U0001f62b",
"\u270c",
"\U0001f60e",
"\U0001f621",
"\U0001f44d",
"\U0001f622",
"\U0001f62a",
"\U0001f60b",
"\U0001f624",
"\u270b",
"\U0001f637",
"\U0001f44f",
"\U0001f440",
"\U0001f52b",
"\U0001f623",
"\U0001f608",
"\U0001f613",
"\U0001f494",
"\u2661",
"\U0001f3a7",
"\U0001f64a",
"\U0001f609",
"\U0001f480",
"\U0001f616",
"\U0001f604",
"\U0001f61c",
"\U0001f620",
"\U0001f645",
"\U0001f4aa",
"\U0001f44a",
"\U0001f49c",
"\U0001f496",
"\U0001f499",
"\U0001f62c",
"\u2728"]

