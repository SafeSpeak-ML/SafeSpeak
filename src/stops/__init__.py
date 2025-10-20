from stop import create_stopwords
from processing import *

custom_stopwords = create_stopwords()
vStoi = None

def set_vocab_stoi(v):
    global vStoi
    vStoi = v.get_stoi()