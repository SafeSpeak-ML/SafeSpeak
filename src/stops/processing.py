import re

from . import custom_stopwords, vStoi
from nltk.tokenize import word_tokenize
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word.lower() for word in tokens if word not in custom_stopwords and word in vStoi] # check word is in vocab or ignore
    return ' '.join(filtered_tokens)
