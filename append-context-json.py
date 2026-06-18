import json

def generate_flat_context_mapping(input_json_path, output_json_path):
    print(f"Loading '{input_json_path}'...")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        nodes = data
    elif isinstance(data, dict):
        nodes = data.values()
    else:
        print("[!] Unrecognized JSON structure.")
        return

    # Temporary dictionary to gather contexts
    temp_map = {}
    
    for node in nodes:
        context = node.get("situational_context", "").strip()
        
        # Skip if missing or errored
        if not context or context.startswith("ERROR"):
            continue
            
        for chunk_id in node.get("source_ids", []):
            if chunk_id not in temp_map:
                temp_map[chunk_id] = []
                
            if context not in temp_map[chunk_id]:
                temp_map[chunk_id].append(context)

    # Build the final array exactly as requested
    final_output = []
    for chunk_id, contexts in temp_map.items():
        # Stitch multiple contexts together into one flat string
        combined_context = " ".join(contexts)
        
        final_output.append({
            "chunk_id": chunk_id,
            "context_to_prepend": combined_context
        })

    # Save to the new JSON file
    print(f"Saving {len(final_output)} items to '{output_json_path}'...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    # Change these filenames to match yours
    INPUT_FILE = "docs/raptor/stella/raptor_level_1_nodes.json" 
    OUTPUT_FILE = "final_contexts.json"
    
    generate_flat_context_mapping(INPUT_FILE, OUTPUT_FILE)