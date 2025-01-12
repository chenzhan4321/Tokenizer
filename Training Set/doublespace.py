import re

def double_space(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace single spaces with double spaces, but not if already double spaced
    double_spaced = re.sub(r'(?<!\s)\s(?!\s)', '  ', content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(double_spaced)

# Assuming the script is in the same directory as result.txt
input_file = 'result.txt'
output_file = 'result_double_spaced.txt'

double_space(input_file, output_file)
print(f"Processed {input_file} and saved result to {output_file}")
