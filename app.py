import torch
import torch.nn as nn
import math
# 7 dictionary size
token_to_ids = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "!": 6
}

# hello world!
input = ["hello", "world", "!", "this"]
input_tokens = [token_to_ids[token] for token in input]

print("input_tokens: ", input_tokens)

embedding = nn.Embedding(7, 3)

print("embedding.weight: ", embedding.weight)

embeddings = embedding(torch.tensor(input_tokens, dtype=torch.long))
print("embeddings: ", embeddings)

# Add positional encoding
d_model = 3 # embedding dimension

def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pe[pos][i] = math.sin(pos / (10000 ** (i/d_model)))
            else:
                pe[pos][i] = math.cos(pos / (10000 ** ((i-1)/d_model)))
    return pe

positional_encoding = get_positional_encoding(len(input_tokens), d_model)
print("positional_encoding: ", positional_encoding)

# Add positional encoding to embeddings
embeddings_with_pe = embeddings + positional_encoding
print("embeddings_with_pe: ", embeddings_with_pe)

# Single-head attention
head_dim = 5

W_q = nn.Linear(d_model, head_dim, bias=False)
W_k = nn.Linear(d_model, head_dim, bias=False)
W_v = nn.Linear(d_model, head_dim, bias=False)

q = W_q(embeddings_with_pe) # Query
k = W_k(embeddings_with_pe) # Key
v = W_v(embeddings_with_pe) # Value

print("q: ", q)
print("k: ", k)
print("v: ", v)

