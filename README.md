# Encoder-Decoder Trainer

This repository contains code for training an BPE model, primarily designed for the tokenization of low-resource languages. The project includes Jupyter notebooks that guide users through the process of training the model and performing encoding and decoding operations.

## Table of Contents

- [Project Description](#project-description)
- [Files in the Repository](#files-in-the-repository)
- [Installation](#installation)
- [Usage](#usage)
- [Vocabulary and Merges](#vocabulary-and-merges)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description

### Overview

The simple BPE program is a Tokenizer. This project provides a hands-on implementation of an BPE model using Python to deal with low-resource languages.

### Features

- **Data Handling**: Includes utilities for preprocessing and handling text data, including tokenization and byte pair encoding (BPE).
- **Jupyter Notebooks**: Interactive notebooks that guide users through the entire process, from data preparation to model training and evaluation.

### Dependencies

- **Python 3.6+**
- **Jupyter Notebook**
- **NumPy**

## Files in the Repository

- `Trainer_0.1.ipynb`: Jupyter notebook for training the encoder-decoder model.
- `En-decoder.ipynb`: Jupyter notebook for encoding and decoding sequences.
- `vocabulary_{vocab_size}.json`: JSON file containing the vocabulary.
- `merges_{vocab_size}.json`: JSON file containing the merges information.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   \`\`\`bash
   git clone https://github.com/chenzhan4321/Tokenizer.git
   \`\`\`

## Usage

### Training the Model

1. Open the `Trainer_0.1.ipynb` notebook in Jupyter:

   \`\`\`bash
   jupyter notebook Trainer_0.1.ipynb
   \`\`\`

2. Follow the instructions in the notebook to train the encoder-decoder model.

### Encoding and Decoding Sequences

1. Open the `En-decoder.ipynb` notebook in Jupyter:

   \`\`\`bash
   jupyter notebook En-decoder.ipynb
   \`\`\`

2. Follow the instructions in the notebook to encode and decode sequences.

## Vocabulary and Merges

The `vocabulary_{vocab_size}.json` file contains the vocabulary used by the model, where each byte value is decoded to a UTF-8 string. The `merges_{vocab_size}.json` file contains the merges information necessary for byte pair encoding.

### Example of Vocabulary Conversion

Here is an example of how the vocabulary is converted:

\`\`\`python
import json

# Load vocabulary
with open('vocabulary_{vocab_size}.json', 'r', encoding='utf-8') as file:
    vocab = json.load(file)

# Display the vocabulary
print(vocab)
\`\`\`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank all the contributors to this project.
