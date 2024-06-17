

# %%
import sentencepiece as spm

# %%
# write a toy.txt file with some random text
with open("toy.txt", "w", encoding="utf-8") as f:
  f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

# %% [markdown]
# Docs for sentencepiece options:
# 
# - [markdown](https://github.com/google/sentencepiece/blob/master/doc/options.md)
# - [protobuf](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L193)

# %%
# train a sentencepiece model on it
# the settings here are (best effort) those used for training Llama 2
import os

options = dict(
  # input spec
  input="./result2.txt",
  input_format="text",
  # output spec
  model_prefix="tok400", # output filename prefix
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=400,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)


# %%
sp = spm.SentencePieceProcessor()
sp.load('tok400.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
vocab

# %%
ids = sp.encode("hello 안녕하세요")
print(ids)

# %%
print([sp.id_to_piece(idx) for idx in ids])

# %% [markdown]
# **Llama 2 tokenizer proto**
# If you'd like to export the raw protocol buffer for the `tokenizer.model` released by meta, this is a [helpful issue](https://github.com/google/sentencepiece/issues/121). And this is the result:
# 
# ```
# normalizer_spec {
#   name: "identity"
#   precompiled_charsmap: ""
#   add_dummy_prefix: true
#   remove_extra_whitespaces: false
#   normalization_rule_tsv: ""
# }
# 
# trainer_spec {
#   input: "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
#   model_prefix: "spm_model_32k_200M_charcov099995_allowWSO__v2"
#   model_type: BPE
#   vocab_size: 32000
#   self_test_sample_size: 0
#   input_format: "text"
#   character_coverage: 0.99995
#   input_sentence_size: 200000000
#   seed_sentencepiece_size: 1000000
#   shrinking_factor: 0.75
#   num_threads: 80
#   num_sub_iterations: 2
#   max_sentence_length: 4192
#   shuffle_input_sentence: true
#   max_sentencepiece_length: 16
#   split_by_unicode_script: true
#   split_by_whitespace: true
#   split_by_number: true
#   treat_whitespace_as_suffix: false
#   split_digits: true
#   allow_whitespace_only_pieces: true
#   vocabulary_output_piece_score: true
#   hard_vocab_limit: true
#   use_all_vocab: false
#   byte_fallback: true
#   required_chars: ""
#   unk_id: 0
#   bos_id: 1
#   eos_id: 2
#   pad_id: -1
#   unk_surface: " \342\201\207 "
#   unk_piece: "<unk>"
#   bos_piece: "<s>"
#   eos_piece: "</s>"
#   pad_piece: "<pad>"
#   train_extremely_large_corpus: false
#   enable_differential_privacy: false
#   differential_privacy_noise_level: 0.0
#   differential_privacy_clipping_threshold: 0
# }
# ```

# %% [markdown]
# #### vocab_size
# 
# - Q: what should be vocab size?
# - Q: how can I increase vocab size?
# - A: let's see. Reminder: [gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) from before.

# %% [markdown]
# ### Final recommendations
# 
# - Don't brush off tokenization. A lot of footguns and sharp edges here. Security issues. Safety issues.
# - Eternal glory to anyone who can delete tokenization as a required step in LLMs.
# - In your own application:
#   - Maybe you can just re-use the GPT-4 tokens and tiktoken?
#   - If you're training a vocab, ok to use BPE with sentencepiece. Careful with the million settings.
#   - Switch to minbpe once it is as efficient as sentencepiece :)
# 

# %% [markdown]
# ### Also worth looking at
# 
# - [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer). I didn't cover it in detail in the lecture because the algorithm (to my knowledge) is very similar to sentencepiece, but worth potentially evaluating for use in practice.


