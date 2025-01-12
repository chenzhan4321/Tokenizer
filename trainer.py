import os
import regex as re
import json
from typing import List, Dict, Tuple

# File reading
def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
# Text tokenization
def word_separation(text: str) -> List[str]:
    # 先用空格分割文本，但保留空格作为token
    tokens = []
    words = text.split()
    
    for i, word in enumerate(words):
        # 先添加一个空格token（除了第一个词之前）
        if i > 0:
            tokens.append(' ')
            
        # 检查并处理词头
        if word and word[0] in "ܕܘܒܡܠܬܐܢ":
            tokens.append(word[0])
            word = word[1:]
            
        # 处理词干和词尾
        for suffix in [
            "ܝܟܝܢ", "ܝܟܘܢ", "ܝܗܝܢ", "ܝܗܘܢ",            # 长度为4的词尾
            "ܟܝܢ", "ܗܝܢ", "ܗܘܢ", "ܟܘܢ", "ܝܟܝ", "ܘܗܝ", "ܢܝ",  # 长度为3的词尾
            "ܝܗ", "ܘܢ", "ܝܢ", "ܝܟ", "ܟܝ", "ܬܐ",          # 长度为2的词尾
            "ܘ", "ܝ", "ܟ", "ܗ", "ܢ", "ܬ", "ܐ"             # 长度为1的词尾
        ]:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem:
                    tokens.append(stem)
                tokens.append(suffix)
                break
        else:  # 如果没有找到词尾
            if word:
                tokens.append(word)
    
    # 如果文本以空格结尾，添加最后的空格
    if text.endswith(' '):
        tokens.append(' ')
    
    return tokens

def encode_words(words: List[str]) -> List[List[int]]:
    encoded = []
    for word in words:
        # 如果是空格，直接编码为特定值（比如32，ASCII中空格的编码）
        if word.isspace():
            encoded.append([32])  # 32 是空格的ASCII码
        else:
            encoded.append(list(map(int, word.encode("utf-8"))))
    return encoded

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
    file_path = os.path.join(current_directory, "Training Set", "result_double_spaced.txt")
    
    print("Reading file...")
    text = read_file(file_path)
    print(f"First 52 characters: {text[:52]}")
    tokens = list(map(int, text.encode("utf-8")))
    print(f"First 12 utf-8 codes: {tokens[:12]}")

    print("\nTokenizing text...")
    text_words = word_separation(text)
    print(f"First 52 words: {text_words[:52]}")
    
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
    vocab[32] = b' '  # 32是空格的ASCII码

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
    vocab_converted = {
        idx: token.decode("utf-8", errors="ignore") 
        for idx, token in vocab.items() 
        if idx > 255 or idx == 32  # 修改这里，让空格(32)也被包含在词汇表中
    }
    
    # 确保空格在词汇表中
    vocab_converted[32] = " "  # 明确添加空格映射
    
    merges_converted = {f"({pair[0]}, {pair[1]})": value for pair, value in merges.items()}

    file_path_voc = os.path.join(current_directory, f"vocabulary_{vocab_size}.json")
    file_path_merge = os.path.join(current_directory, f"merges_{vocab_size}.json")

    print("Saving vocabulary and merges...")
    with open(file_path_voc, "w", encoding="utf-8") as file:
        json.dump(vocab_converted, file, indent=4, ensure_ascii=False)
    with open(file_path_merge, "w", encoding="utf-8") as file:
        json.dump(merges_converted, file, indent=4)

    print(f"Vocabulary saved to: {file_path_voc}")
    print(f"Merges saved to: {file_path_merge}")

    # 添加示例文本的tokenization演示
    print("\n=== Tokenization Examples ===")
    example_texts = [
        "ܫܠܡܐ ܠܟ",  # "Hello to you"
        "ܒܝܬܐ ܪܒܐ",  # "Big house"
        "ܐܢܐ ܐܙܠ ܠܒܝܬܐ"  # "I go home"
    ]
    
    print("\nShowing how each text is tokenized:")
    for text in example_texts:
        print(f"\nOriginal text: {text}")
        words = word_separation(text)
        print(f"Words: {words}")
        encoded = encode_words(words)
        print(f"Initial encoding: {encoded}")
        
        # 应用BPE合并
        current_ids = list(encoded)
        for pair, idx in merges.items():
            current_ids = merge(current_ids, pair, idx)
        
        print(f"Final encoding: {current_ids}")
        
        # 解码并显示最终的token
        final_tokens = []
        for id_list in current_ids:
            for id in id_list:
                if id > 255:
                    final_tokens.append(vocab_converted[id])
                else:
                    final_tokens.append(vocab[id].decode('utf-8', errors='ignore'))
        
        print(f"Final tokens: {final_tokens}")

if __name__ == "__main__":
    main()
