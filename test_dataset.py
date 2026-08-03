from datasets import load_dataset

dataset = load_dataset(
    "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
)

sample = dataset["train"][0]

print("keys:")
print(sample.keys())

print("\nops:")
print(sample["ops"])

print("\ncode:")
print(sample["code"])
