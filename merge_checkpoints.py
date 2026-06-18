import os
import json
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Merge RAPTOR checkpoint files into a single JSON.")
    parser.add_argument("--target-level", type=int, required=True, help="The level of checkpoints to merge (e.g., 1, 2)")
    parser.add_argument("--base-dir", type=str, default="docs/raptor/stella", help="Base directory where checkpoints are stored")
    args = parser.parse_args()

    # Construct paths based on your existing architecture
    checkpoint_dir = os.path.join(args.base_dir, f"level_{args.target_level}_checkpoints")
    output_file = os.path.join(args.base_dir, f"raptor_level_{args.target_level}_nodes.json")

    print("="*60)
    print(f" [RAPTOR] Merging Checkpoints for Level {args.target_level} ")
    print("="*60)

    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Directory not found: {checkpoint_dir}")
        print("Make sure you entered the correct target level and base directory.")
        return

    # Find all valid JSON files (using 'cluster_*.json' ignores your 'ERROR_*.txt' logs)
    search_pattern = os.path.join(checkpoint_dir, "cluster_*.json")
    checkpoint_files = glob.glob(search_pattern)

    if not checkpoint_files:
        print(f"[WARNING] No checkpoint files found in {checkpoint_dir}")
        return

    print(f" -> Found {len(checkpoint_files)} checkpoint files. Merging...")

    merged_data = []
    error_count = 0

    for file_path in checkpoint_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                node_data = json.load(f)
                merged_data.append(node_data)
        except Exception as e:
            print(f"[ERROR] Failed to read {os.path.basename(file_path)}: {e}")
            error_count += 1

    # Save to the final merged file
    print(f"\n -> Writing merged data to '{output_file}'...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Could not write output file: {e}")
        return

    print("\n" + "="*60)
    print(" [SUCCESS] Checkpoints Merged Successfully! ")
    print("="*60)
    print(f" Total Nodes Merged : {len(merged_data)}")
    if error_count > 0:
        print(f" Files Failed       : {error_count}")
    print(f" Output File        : {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()