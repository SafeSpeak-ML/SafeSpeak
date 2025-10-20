import pickle
import time
import torch

from concurrent.futures import ThreadPoolExecutor 
from torchtext.vocab import Vocab

# locals
from stops import *
from classifier import TextClassifier

# setup the actual model from pickle
with open('../model/vocab.pkl', 'rb') as f:
    vocab: Vocab = pickle.load(f)
vocab_size = len(vocab)

# allow async
executor = ThreadPoolExecutor(max_workers=4)

# Model parameters (matching training)
embedding_dim = 100
hidden_dim = 128
num_classes = 2

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load("../model/dectection.pth", map_location=device))
model.to(device)

set_vocab_stoi(vocab)

# prediction actual logic
def predict_text(text, model=model, tokenizer=tokenizer, vocab=vocab, device=device):
    model.eval()
    tokens = tokenizer(text)
    indices = [vocab[token] for token in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

async def predict(text):
    text = preprocess_text(text)
    if len(text.strip()) < 50: # min len check
        return (None, 0.0)
    start = time.time()*1000
    result = executor.submit(predict_text, text).result() # make it async
    end = time.time()*1000
    return (False, end - start) if result == 0 else (True, end - start)