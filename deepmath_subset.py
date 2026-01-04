import argparse
import os
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/shared/nlp_workspace/verl/data/deepmath_subset")
    parser.add_argument("--sample_size", type=int, default=7000, help="Number of examples to keep after filtering")
    args = parser.parse_args()

    # Load dataset
    deepmath = load_dataset("zwhe99/DeepMath-103K")["train"]

    # Filter by difficulty (5–10 range)
    deepmath = deepmath.filter(lambda ex: 5 <= ex["difficulty"] <= 10)

    # Shuffle and take a subset of ~7k examples
    # deepmath = deepmath.shuffle(seed=42)
    if args.sample_size and args.sample_size < len(deepmath):
        deepmath = deepmath.select(range(args.sample_size))

    # Split into train/test (95/5)
    split_dataset = deepmath.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # Define format conversion
    def convert(split):
        def process_fn(example, idx):
            question = example.pop("question", "").strip()
            final_answer = example.pop("final_answer", "").strip()
            return {
                "data_source": "deepmath_subset",
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            question
                            + "\n\n"
                            + "Please reason step by step, and put your final answer within \\boxed{}"
                        ),
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": final_answer},
                "extra_info": {"index": idx, "split": split},
            }

        return process_fn

    # Apply mapping
    train_dataset = train_dataset.map(function=convert("train"), with_indices=True)
    test_dataset = test_dataset.map(function=convert("test"), with_indices=True)

    # Save locally
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    print(f"✅ Saved filtered subset to {args.local_dir}")
    print(f"   Train: {len(train_dataset)} | Test: {len(test_dataset)}")
