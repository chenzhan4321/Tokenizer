
# %%
import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to the file
file_path = os.path.join(current_directory, "Training Set", "result.txt")

# Open the file and read its contents
with open(file_path, "r") as file:
    text = file.read()


tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print (text[:52])
print (tokens[:52])




# %%
import regex as re
pattern = re.compile(r""" ?ܘ(?=\p{L}+)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# This regex pattern is used to tokenize the text into words and other meaningful units
# Here's a breakdown of the pattern:

# ܘ(?=\p{L}+)     : Matches 'ܘ' (Syriac Waw) only if followed by one or more letters
#                   This is likely for a specific linguistic feature in Syriac

# \p{L}+          : Matches one or more Unicode letters

# \p{N}+          : Matches one or more Unicode numbers

# [^\s\p{L}\p{N}]+: Matches one or more characters that are not whitespace, letters, or numbers

# \s+(?!\S)       : Matches one or more whitespace characters not followed by a non-whitespace character
#                   This is likely to catch trailing spaces

# \s+             : Matches one or more whitespace characters

# Each part of the pattern is preceded by a space and a question mark ( ?)
# This makes the leading space optional for each token

# The pipe symbol (|) separates each part of the pattern, allowing any of these to match



text_words = re.findall(pattern, text) # devide the text into words according to gpt2pat pattern
print(text_words[:52])
print(type(text_words))
words_tokens = []
for i in text_words:
    token_word = i.encode("utf-8")
    token = list(map(int, token_word))
    words_tokens.append(token)
print(words_tokens[:3])



# %%
def get_stats(ids):
    counts = {}
    for id in ids:    
        for pair in zip(id, id[1:]): # Pythonic way to iterate consecutive elements
            if pair[1] != 220:  # Check to avoid pairs ending with 220
                counts[pair] = counts.get(pair, 0) + 1
    return counts


def simple_get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        if pair[1] != ord(' ') and pair[1] != 220:  # Check to avoid pairs ending with 220
            counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(words_tokens)
top_pair = max(stats, key=stats.get)
print (top_pair, stats[top_pair])

# %%
def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  for sublist in ids:
        i = 0
        new_sublist = []
        while i < len(sublist):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(sublist) - 1 and sublist[i] == pair[0] and sublist[i + 1] == pair[1]:
                new_sublist.append(idx)
                i += 2
            else:
                new_sublist.append(sublist[i])
                i += 1
        newids.append(new_sublist)
  return newids

def simple_merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def length(ids):
    return sum(len(id) for id in ids)

print("length:", length(words_tokens))
tokens2 = merge(words_tokens, top_pair, 256)
print(tokens2[:52])
print("length:", length(tokens2))

# %%


# ---
vocab_size = 6000 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(words_tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  ids = merge(ids, pair, idx)
  merges[pair] = idx

# %%
print("tokens length:", length(words_tokens))
print("ids length:", length(ids))
print(f"compression ratio: {length(words_tokens) / length(ids):.2f}X")

# %% [markdown]
# ### Generate Vocab
# 

# %%
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

vocab_converted= {idx: vocab[idx].decode("utf-8") for idx in vocab if idx > 255}

merges_converted = {str(key): value for key, value in merges.items()}

# %%
import json


file_path_voc = os.path.join(current_directory, f"vocabulary_{vocab_size}.json")
file_path_merge = os.path.join(current_directory, f"merges_{vocab_size}.json")


with open(file_path_voc, "w", encoding="utf-8") as file:
    json.dump(vocab_converted, file, indent=4, ensure_ascii=False)
with open(file_path_merge, "w") as file:
    json.dump(merges_converted, file, indent=4)   

    
print(f"Vocabulary and merges saved to {file_path}")


