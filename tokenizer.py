import json
import os
import regex as re
from typing import List, Dict, Tuple, Optional

# 定义类型别名
TokenId = int  # token ID的类型
TokenPair = Tuple[int, int]  # token对的类型
MergeDict = Dict[TokenPair, TokenId]  # 合并规则字典的类型
VocabDict = Dict[TokenId, str]  # 词汇表字典的类型

# 定义配置常量
SPACE_TOKEN = 32  # 空格token的值
UNKNOWN_TOKEN_FORMAT = "<{0}>"  # 未知token的格式
DEFAULT_SAMPLE_SIZE = 50  # 默认样本大小
DEFAULT_ENCODING = "utf-8"  # 默认编码

# 定义叙利亚文常量
PREFIXES = "ܕܘܒܡܠܬܐܢ"  # 叙利亚文词头
# 定义叙利亚文词尾，按长度降序排序
SUFFIXES = sorted([
    "ܝܟܝܢ", "ܝܟܘܢ", "ܝܗܝܢ", "ܝܗܘܢ",            # 长度为4的词尾
    "ܟܝܢ", "ܗܝܢ", "ܗܘܢ", "ܟܘܢ", "ܝܟܝ", "ܘܗܝ", "ܢܝ",  # 长度为3的词尾
    "ܝܗ", "ܘܢ", "ܝܢ", "ܝܟ", "ܟܝ", "ܬܐ",          # 长度为2的词尾
    "ܘ", "ܝ", "ܟ", "ܗ", "ܢ", "ܬ", "ܐ"             # 长度为1的词尾
], key=len, reverse=True)

def load_merges(file_path: str) -> MergeDict:
    """从JSON文件加载合并规则"""
    try:
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            merges_data = json.load(f)
            # 将字符串键转换为元组键
            return {
                tuple(map(int, k.strip('()').split(','))): v 
                for k, v in merges_data.items()
            }
    except FileNotFoundError:
        # 文件不存在时抛出异常
        raise FileNotFoundError(f"合并规则文件未找到: {file_path}")
    except json.JSONDecodeError:
        # JSON格式无效时抛出异常
        raise ValueError(f"合并规则文件包含无效的JSON格式: {file_path}")
    except Exception as e:
        # 其他错误时抛出异常
        raise ValueError(f"处理合并规则文件时出错: {str(e)}")

def load_vocab(file_path: str) -> VocabDict:
    """从JSON文件加载词汇表
    
    Args:
        file_path: 词汇表JSON文件的路径
        
    Returns:
        包含token ID到字符串映射的字典
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式无效
        ValueError: 数据格式不正确
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            return {int(k): v for k, v in vocab_data.items()}
    except FileNotFoundError:
        raise FileNotFoundError(f"词汇表文件未找到: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"词汇表文件包含无效的JSON格式: {file_path}")
    except Exception as e:
        raise ValueError(f"处理词汇表文件时出错: {str(e)}")

def word_separation(text: str) -> List[str]:
    """将叙利亚文本分割为词、词头和词尾的列表
    
    Args:
        text: 输入的叙利亚文本
        
    Returns:
        包含分词结果的列表，每个元素可能是:
        - 空格
        - 词头 (来自PREFIXES)
        - 词干
        - 词尾 (来自SUFFIXES)
        
    Note:
        - 每个词都会被空格分隔
        - 如果词以PREFIXES中的字符开始，会被分离出来
        - 会尝试匹配最长的可能词尾
    """
    words = text.split()
    tokens = []
    
    for word in words:
        # 添加空格和可能的词头
        if word and word[0] in PREFIXES:
            tokens.extend([' ', word[0]])
            word = word[1:]
        else:
            tokens.append(' ')
            
        # 处理词干和词尾
        for suffix in SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem:
                    tokens.append(stem)
                tokens.append(suffix)
                break
        else:  # 没有找到词尾
            if word:
                tokens.append(word)
                
    print(tokens)
    return tokens


def encode_words(words: List[str]) -> List[List[TokenId]]:
    """将词列表编码为整数列表
    
    Args:
        words: 输入词列表
        
    Returns:
        编码后的整数列表的列表，其中:
        - 空格被编码为[32]
        - 其他词被编码为UTF-8字节值列表
    """
    encoded = []
    for word in words:
        if word==" ":  # 如果是空格
            encoded.append([32])  # 直接使用32作为空格token
        else:
            encoded.append(list(map(int, word.encode("utf-8"))))

    return encoded

def merge(token: List[TokenId], merges: MergeDict, vocab: VocabDict) -> List[TokenId]:
    """使用merges字典合并token中的相邻对
    
    Args:
        token: 要处理的整数token列表
        merges: 合并规则字典，键为token对，值为合并后的token
        vocab: 词汇表字典，用于验证合并结果
        
    Returns:
        合并后的token列表
        
    Note:
        - 会持续合并直到没有可以合并的相邻对
        - 合并顺序从左到右
    """
    if len(token) <= 1:
        return token
        
    while True:
        # 找到第一个可以合并的对
        for i, pair in enumerate(zip(token, token[1:])):
            if pair in merges:
                token = token[:i] + [merges[pair]] + token[i+2:]
                break
        else:  # 没有找到可以合并的对
            break
            
    return token

def decode_token(token: List[TokenId], vocab: VocabDict) -> str:
    """将整数token解码为字符串
    
    Args:
        token: 要解码的整数token列表
        vocab: 词汇表字典，将token ID映射到字符串
        
    Returns:
        解码后的字符串，未知的token会被替换为<token_id>格式
    """
    result = []
    for t in token:
        if t == 32:  # 如果是空格token
            result.append(' ')
        elif t > 255:  # 如果是BPE合并后的token
            result.append(vocab.get(t, UNKNOWN_TOKEN_FORMAT.format(t)))
        else:  # 如果是基础UTF-8编码
            result.append(bytes([t]).decode('utf-8', errors='ignore'))
    return ''.join(result)

def validate_files(*file_paths: str) -> Optional[List[str]]:
    """验证文件是否存在
    
    Args:
        file_paths: 要验证的文件路径
        
    Returns:
        如果有文件不存在，返回不存在的文件列表；否则返回None
    """
    missing = [f for f in file_paths if not os.path.exists(f)]
    return missing if missing else None

def tokenize_file(input_file: str, merges_file: str, vocab_file: str, output_file: str) -> None:
    """处理输入文件并将分词结果写入输出文件
    
    Args:
        input_file: 输入文本文件路径
        merges_file: 合并规则JSON文件路径
        vocab_file: 词汇表JSON文件路径
        output_file: 输出文件路径
        
    Raises:
        FileNotFoundError: 任何所需文件不存在
        ValueError: 输入文件格式错误
        Exception: 其他处理错误
    """
    # 验证输入文件
    missing_files = validate_files(input_file, merges_file, vocab_file)
    if missing_files:
        raise FileNotFoundError(
            "以下文件不存在:\n" + "\n".join(f"  - {f}" for f in missing_files)
        )
    try:
        # 加载必要的数据
        merges = load_merges(merges_file)
        vocab = load_vocab(vocab_file)
        
        # 读取和处理文本
        with open(input_file, 'r', encoding=DEFAULT_ENCODING) as f:
            text = f.read()
        
        # 分词处理
        words = word_separation(text)
        encoded_words = encode_words(words)
        tokenized = [merge(token, merges, vocab) for token in encoded_words]
        decoded_tokens = [decode_token(token, vocab) for token in tokenized]
        
        # 写入结果
        output_text = '.'.join(decoded_tokens)
        with open(output_file, 'w', encoding=DEFAULT_ENCODING) as f:
            f.write(output_text)
        
        # 打印样本
        sample_size = min(DEFAULT_SAMPLE_SIZE, len(decoded_tokens))
        print(tokenized[:50])
        print(f"\nTokenized text saved to {output_file}")
        print(f"Sample of first {sample_size} tokens:")
        print('.'.join(decoded_tokens[:sample_size]))
        
    except Exception as e:
        raise RuntimeError(f"分词处理失败: {str(e)}")

if __name__ == "__main__":
    try:
        current_directory = os.getcwd()
        merges_file = os.path.join(current_directory, "merges_500.json")
        vocab_file = os.path.join(current_directory, "vocabulary_500.json")
        input_file = os.path.join(current_directory, "test.txt")
        output_file = os.path.join(current_directory, "tokenized_output.txt")
        
        # 验证所需文件
        missing_files = validate_files(merges_file, vocab_file, input_file)
        if missing_files:
            print("错误: 以下文件不存在:\n" + 
                  "\n".join(f"  - {f}" for f in missing_files))
            exit(1)
            
        tokenize_file(input_file, merges_file, vocab_file, output_file)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)
