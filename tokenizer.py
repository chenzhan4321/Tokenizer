import json
import os
import regex as re
from typing import List, Dict, Tuple

def load_merges(file_path: str) -> Dict[Tuple[int, int], int]:
    with open(file_path, 'r') as f:
        merges_data = json.load(f)
    return {tuple(map(int, k.strip('()').split(','))): v for k, v in merges_data.items()}

def load_vocab(file_path: str) -> Dict[int, str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return {int(k): v for k, v in vocab_data.items()}

def tokenize_text(text: str) -> List[str]:
    pattern = re.compile(r""" ?ܘ(?=\p{L}+)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return re.findall(pattern, text)

def encode_words(words: List[str]) -> List[List[int]]:
    return [list(map(int, word.encode("utf-8"))) for word in words]

def merge(token: List[int], merges: Dict[Tuple[int, int], int], vocab: Dict[int, str]) -> List[int]:
    while len(token) > 1:
        pairs = list(zip(token, token[1:]))
        if not any((pair in merges) for pair in pairs):
            break

        for i, pair in enumerate(pairs):
            if pair in merges:
                new_token = merges[pair]
                print(f"Merging: {pair} ({vocab.get(pair[0], '<unk>')}{vocab.get(pair[1], '<unk>')}) -> {new_token} ({vocab.get(new_token, '<unk>')})")
                token = token[:i] + [new_token] + token[i+2:]
                break
    return token

def decode_token(token: List[int], vocab: Dict[int, str]) -> str:
    return ''.join(vocab.get(t, f'<{t}>') for t in token)

def tokenize_file(input_file: str, merges_file: str, vocab_file: str, output_file: str):
    merges = load_merges(merges_file)
    vocab = load_vocab(vocab_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    words = tokenize_text(text)
    encoded_words = encode_words(words)
    tokenized = [merge(token, merges, vocab) for token in encoded_words]
    
    decoded_tokens = [decode_token(token, vocab) for token in tokenized]
    output_text = '.'.join(decoded_tokens)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\nTokenized text saved to {output_file}")
    print(f"Sample of tokenized text (first 10 tokens):")
    print('.'.join(decoded_tokens[:10]))

if __name__ == "__main__":
    current_directory = os.getcwd()
    merges_file = os.path.join(current_directory, "merges_500.json")
    vocab_file = os.path.join(current_directory, "vocabulary_500.json")
    input_file = os.path.join(current_directory, "test.txt")
    output_file = os.path.join(current_directory, "tokenized_output.txt")
    
    tokenize_file(input_file, merges_file, vocab_file, output_file)