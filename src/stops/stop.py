from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def create_stopwords():
    all_words = []
    texts = []
    with open('./texts/general.txt', 'r') as f:
        texts.append(f.readlines())
    with open('./texts/slang.txt', 'r') as f:
        texts.append(f.readlines())
    texts = set(texts) # faster search
    for text in texts:
        word = word_tokenize(text.lower())
        all_words.extend(word)
    
    word_freq = Counter(all_words)
    custom_stops = {word for word in word_freq.keys()}
    return custom_stops.union(set(stopwords.words('english')))