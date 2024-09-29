from datasets import load_dataset


ds = load_dataset("Maxscha/commitbench", split="train")
print(ds)
ds[100]