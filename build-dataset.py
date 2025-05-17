import json
import os
import argparse
from pathlib import Path

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dataset(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def insert_character_periodically(knowledge_data, character_data, interval=50):
    result = []
    for i in range(0, len(knowledge_data), interval):
        block = knowledge_data[i:i+interval]
        result.extend(block)
        if character_data:
            result.extend(character_data)
    return result

def build_dataset(dataset_dir, output_file, interval=50):
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "__zeta__.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    knowledge_data = []
    character_data = []

    for name, config in meta["datasets"].items():
        file_path = dataset_dir / config["file"]
        data = load_dataset(file_path)

        if config["type"] == "knowledge":
            knowledge_data.extend(data)
        elif config["type"] == "character":
            character_data.extend(data)

    mixed_data = insert_character_periodically(knowledge_data, character_data, interval=interval)

    save_dataset(output_file, mixed_data)
    print(f"Dataset built successfully: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AzukiF dataset from Zeta-Dataset Package (Zeta-2 or Later)")
    parser.add_argument("dataset_dir", help="Path to dataset directory (containing __zeta__.json)")
    parser.add_argument("output_file", help="Path to output AzukiF file (e.g., merged.json)")
    parser.add_argument("--interval", type=int, default=40, help="Insert character data every N knowledge samples")
    args = parser.parse_args()

    build_dataset(args.dataset_dir, args.output_file, interval=args.interval)
