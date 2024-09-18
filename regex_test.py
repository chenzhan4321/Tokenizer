import regex

# 正则表达式
#pattern = r'\b[ܘܒܡܠ](?=\p{L}+)'
pattern = regex.compile(r""" [ܘܒܡܠ](?=\p{L}+)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# 示例文本
text = "ܡܠܟܐ ܒܘܬ ܘܡܠܟܐ ܒܐܘܒ"

# 查找匹配
matches = regex.findall(pattern, text)

# 输出结果
print(matches)