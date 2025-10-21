"""Compares the number of tokens produced by different tokenizers."""

# https://huggingface.co/datasets/cerebras/SlimPajama-627B
from datasets import load_dataset

ds = load_dataset("cerebras/SlimPajama-627B", streaming=True)
slimpajama_train, slimpajama_test = ds["train"], ds["test"]


texts = []

for idx, item in enumerate(slimpajama_train):
    print(item)
    texts.append(item["text"])
    if idx > 5:
        break


# tiktoken
import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
# tiktoken.encoding_for_model("gpt-4o")

# cl100k_base
tokenizer = tiktoken.get_encoding("cl100k_base")  # used for gpt-3.5 ad gpt-4
# o200k_harmony
tokenizer2 = tiktoken.get_encoding("o200k_base")  # gpt-oss


# measure the number of tokens in the dataset with both tokenizers
num_tokens = 0
num_tokens2 = 0
for text in texts:
    num_tokens += len(tokenizer.encode(text))
    num_tokens2 += len(tokenizer2.encode(text))
    # if idx % 100 == 0:
    #     print(f"Processed {idx} samples")
    # if idx > 1000:
    #     break
print(f"Number of tokens: {num_tokens}")
print(f"Number of tokens2: {num_tokens2}")
