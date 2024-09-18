import os
import regex as re
import json
from typing import List, Dict, Tuple

# File reading
def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Text tokenization
def tokenize_text(text: str) -> List[str]:
    # Compile a regex pattern for tokenization
    pattern = re.compile(r""" ?ܘ(?=\p{L}+)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    tokens = re.findall(pattern, text)
    return [token for token in tokens]

def encode_words(words: List[str]) -> List[List[int]]:
    return [list(map(int, word.encode("utf-8"))) for word in words]

# BPE Core Functions
def get_stats(ids: List[List[int]]) -> Dict[Tuple[int, int], int]:
    counts = {}
    for id_list in ids:
        for pair in zip(id_list, id_list[1:]):
            if pair[1] != 220:  # Avoid pairs ending with 220 (Syriac letters)
                counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[List[int]], pair: Tuple[int, int], idx: int) -> List[List[int]]:
    newids = []
    for sublist in ids:
        i = 0
        new_sublist = []
        while i < len(sublist):
            if i < len(sublist) - 1 and sublist[i] == pair[0] and sublist[i + 1] == pair[1]:
                new_sublist.append(idx)
                i += 2
            else:
                new_sublist.append(sublist[i])
                i += 1
        newids.append(new_sublist)
    return newids

def calculate_length(ids: List[List[int]]) -> int:
    return sum(len(id_list) for id_list in ids)

# Main execution
def main():
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "Training Set", "result.txt")
    
    print("Reading file...")
    text = read_file(file_path)
    print(f"First 52 characters: {text[:52]}")
    tokens = list(map(int, text.encode("utf-8")))
    print(f"First 12 utf-8 codes: {tokens[:12]}")

    print("\nTokenizing text...")
    text_words = tokenize_text(text)
    print(f"First 52 word tokens: {text_words[:52]}")
    
    print("\nEncoding words...")
    words_tokens = encode_words(text_words)
    print(f"First 30 encoded tokens: {words_tokens[:30]}")

    print(f"\nInitial length: {calculate_length(words_tokens)}")

    vocab_size = 500
    print(f"\nTraining BPE with vocabulary size {vocab_size}...")
    num_merges = vocab_size - 256
    ids = list(words_tokens)
    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"Merge {i+1}: pair {pair} -> {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    print(f"\nFinal length: {calculate_length(ids)}")
    print(f"Compression ratio: {calculate_length(words_tokens) / calculate_length(ids):.2f}X")

    # Generate and save vocabulary and merges
    print("\nGenerating vocabulary and merges...")
    vocab_converted = {idx: token.decode("utf-8", errors="ignore") for idx, token in vocab.items() if idx > 255}
    merges_converted = {str(key): value for key, value in merges.items()}

    file_path_voc = os.path.join(current_directory, f"vocabulary_{vocab_size}.json")
    file_path_merge = os.path.join(current_directory, f"merges_{vocab_size}.json")

    print("Saving vocabulary and merges...")
    with open(file_path_voc, "w", encoding="utf-8") as file:
        json.dump(vocab_converted, file, indent=4, ensure_ascii=False)
    with open(file_path_merge, "w", encoding="utf-8") as file:
        json.dump(merges_converted, file, indent=4)

    print(f"Vocabulary saved to: {file_path_voc}")
    print(f"Merges saved to: {file_path_merge}")

if __name__ == "__main__":
    main()

