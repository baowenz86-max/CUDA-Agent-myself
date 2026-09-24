from datasets import load_dataset

dataset = load_dataset(
    "BytedTsinghua-SIA/CUDA-Agent-Ops-6K",
    cache_dir="./datasets"
)

print(dataset)

print(dataset["train"][0])
