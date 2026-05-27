#!/usr/bin/env python3
"""
GraphRAG extraction pipeline for processing summarized text nodes into a knowledge graph.
Uses Level 2 clusters as batches with their Level 1 nodes as context.
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime

import openai
from tqdm import tqdm

# ============================================================================
# SYSTEM PROMPT - BATCHED UNIVERSAL GRAPH EXTRACTION
# ============================================================================

BATCHED_UNIVERSAL_GRAPH_PROMPT = """
You are an advanced data extraction algorithm tasked with building a Universal Knowledge Graph. Your objective is to extract entities, relationships, and claims from the provided text. You must output strictly valid JSON. Do not include markdown, conversational text, or explanations.

### INPUT STRUCTURE ###
You will receive:
1. A MACRO CONTEXT section containing the Level 2 cluster summary of a RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system
2. Multiple Level 1 nodes, each with Context and Summary fields

Extract entities, relationships, and claims from the COMBINED content of all sections.
Use the MACRO CONTEXT to inform your understanding of entities that appear across multiple Level 1 nodes.

You must categorize all extracted entities into exactly one of these types:
- ACTOR: A person, organization, faction, or autonomous entity capable of intent.
- CONCEPT: An abstract idea, theory, philosophy, methodology, or ruleset.
- ARTIFACT: A physical or digital object, tool, document, or piece of technology.
- EVENT: A specific historical occurrence, milestone, or battle fixed in time.
- LOCATION: A physical, digital, or conceptual space.

Your JSON output must contain exactly three arrays: "entities", "relationships", and "claims".

1. "entities": An array of objects, each containing:
   - "id": A unique, capitalized name for the entity.
   - "type": One of the five allowed types above.
   - "description": A concise, 15-25 word description of the entity based ONLY on the text.

2. "relationships": An array of objects defining how entities interact, containing:
   - "source": The ID of the originating entity.
   - "target": The ID of the receiving entity.
   - "type": A capitalized, snake_case verb (e.g., FOUNDED, DEFEATED_BY, UTILIZES, CONTRADICTS).
   - "description": A brief explanation of how they relate.

3. "claims": An array of objects capturing subjective statements, debates, or lore assertions, containing:
   - "subject": The entity making the claim.
   - "claim": The specific assertion or argument being made.
   - "object": The entity the claim is about.
   - "status": The apparent truthfulness of the claim based on the text (e.g., TRUE, FALSE, THEORIZED, DISPUTED, PROPAGANDA).

### EXAMPLE OUTPUT ###
{
  "entities": [
    {"id": "Bruce Lee", "type": "ACTOR", "description": "A legendary martial artist and founder of a hybrid combat philosophy."},
    {"id": "Jeet Kune Do", "type": "CONCEPT", "description": "A formless, adaptable martial arts philosophy prioritizing efficiency."},
    {"id": "Traditional Karate", "type": "CONCEPT", "description": "A rigid, form-based martial art system."},
    {"id": "1964 Long Beach Championships", "type": "EVENT", "description": "A martial arts tournament where new philosophies were demonstrated."}
  ],
  "relationships": [
    {"source": "Bruce Lee", "target": "Jeet Kune Do", "type": "FOUNDED", "description": "Created the philosophy to break away from rigid styles."},
    {"source": "Bruce Lee", "target": "1964 Long Beach Championships", "type": "DEMONSTRATED_AT", "description": "Showcased his new techniques to the martial arts community."}
  ],
  "claims": [
    {"subject": "Bruce Lee", "claim": "Rigid forms limit a fighter's ability to adapt in real combat.", "object": "Traditional Karate", "status": "THEORIZED"}
  ]
}
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

LEVEL2_FILE = "docs/raptor/stella/raptor_level_2_nodes.json"
LEVEL1_FILE = "docs/raptor/stella/raptor_level_1_nodes.json"
OUTPUT_DIR = "docs/graph_checkpoints"
DEBUG_DIR = "docs/graph_debug"  # For saving batch texts for inspection

# API Configuration - Load from .env file or environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("python-dotenv not installed. Using environment variables only.")
    print("Install with: pip install python-dotenv")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120.0"))  # Default 2 minute timeout


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG extraction pipeline using Level 2 clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_graph_glm.py                    # Run full extraction
  python extract_graph_glm.py -i                # Inspect first batch text
  python extract_graph_glm.py --dry-run         # Show what would be processed
  python extract_graph_glm.py --resume          # Skip already processed clusters
  python extract_graph_glm.py --delay 2.0       # Add 2 second delay between API calls
        """
    )
    
    parser.add_argument(
        "-i", "--inspect",
        action="store_true",
        help="Save first batch text to a .txt file for inspection and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which clusters would be processed without making API calls"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clusters that have already been processed"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between API calls (default: 0.0)"
    )
    
    parser.add_argument(
        "--cluster-id",
        type=str,
        help="Process only a specific cluster ID"
    )
    
    parser.add_argument(
        "--max-clusters",
        type=int,
        help="Maximum number of clusters to process"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=OPENAI_TIMEOUT,
        help=f"API timeout in seconds (default: {OPENAI_TIMEOUT})"
    )
    
    return parser.parse_args()


def load_level1_nodes(level1_file: str) -> List[Dict[str, Any]]:
    """Load the Level 1 summarized text nodes from JSON file."""
    try:
        with open(level1_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not load Level 1 nodes from {level1_file}: {e}")
        sys.exit(1)


def load_level2_clusters(level2_file: str) -> List[Dict[str, Any]]:
    """Load the Level 2 cluster nodes from JSON file."""
    try:
        with open(level2_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not load Level 2 clusters from {level2_file}: {e}")
        sys.exit(1)


def create_directories(output_dir: str, debug_dir: str) -> None:
    """Create output and debug directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(debug_dir).mkdir(parents=True, exist_ok=True)


def get_processed_clusters(output_dir: str) -> Set[str]:
    """Get set of already processed cluster IDs."""
    processed = set()
    if not Path(output_dir).exists():
        return processed
    
    for file_path in Path(output_dir).glob("cluster_*.json"):
        cluster_id = file_path.stem.replace("cluster_", "")
        processed.add(cluster_id)
    
    return processed


def format_level1_node_text(node: Dict[str, Any], cluster_id: str, node_id: str) -> str:
    """Format a single Level 1 node into readable text."""
    parts = [f"Level 1 Node {node_id} from Cluster {cluster_id}:"]
    
    if "situational_context" in node and node["situational_context"]:
        parts.append(f"Context: {node['situational_context']}")
    
    if "cluster_summary" in node:
        parts.append(f"Summary: {node['cluster_summary']}")
    
    return "\n".join(parts)


def process_cluster_batch(level2_cluster: Dict[str, Any], level1_nodes: List[Dict[str, Any]]) -> str:
    """
    Combine Level 2 cluster with its Level 1 nodes into a single text block.
    
    Structure:
    - MACRO CONTEXT (Level 2 summary) at the top
    - Followed by all associated Level 1 nodes
    """
    cluster_id = level2_cluster.get("cluster_id", "unknown")
    source_ids = level2_cluster.get("source_ids", [])
    level2_summary = level2_cluster.get("cluster_summary", "")
    
    batch_text_parts = [
        f"MACRO CONTEXT FOR THIS BATCH: {level2_summary}",
        f"\nThis batch contains {len(source_ids)} Level 1 nodes from Cluster {cluster_id}:\n"
    ]
    
    # Look up and add each Level 1 node
    for source_id in source_ids:
        # source_id is a string representing the node ID (1-154)
        try:
            node_idx = int(source_id) - 1  # Convert to 0-based index
            if 0 <= node_idx < len(level1_nodes):
                node = level1_nodes[node_idx]
                batch_text_parts.append(format_level1_node_text(node, cluster_id, source_id))
                batch_text_parts.append("")  # Empty line between nodes
            else:
                print(f"Warning: Level 1 node index {node_idx} out of range for cluster {cluster_id}")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process source_id '{source_id}' in cluster {cluster_id}: {e}")
    
    return "\n".join(batch_text_parts)


def save_batch_text_for_inspection(batch_text: str, cluster_id: str, debug_dir: str) -> None:
    """Save batch text to a file for inspection."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cluster_{cluster_id}_batch_text_{timestamp}.txt"
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(batch_text)
    
    print(f"Batch text saved to {filepath}")
    print(f"Batch text length: {len(batch_text)} characters")


def call_llm_api(batch_text: str, client: openai.OpenAI, timeout: float = OPENAI_TIMEOUT, max_retries: int = 3) -> Dict[str, Any]:
    """Call the LLM API to extract graph data from the batch with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,  # Replace with your model name if different
                messages=[
                    {"role": "system", "content": BATCHED_UNIVERSAL_GRAPH_PROMPT},
                    {"role": "user", "content": batch_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=timeout
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received empty response from LLM API")
            return json.loads(content)
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  API call failed after {max_retries} attempts: {e}")
                raise
    
    # This should never be reached due to the raise in the else clause above
    raise RuntimeError("Unexpected error: max_retries exhausted without raising exception")


def save_cluster_result(result: Dict[str, Any], cluster_id: str, output_dir: str) -> None:
    """Save cluster result as a separate JSON file."""
    filename = f"cluster_{cluster_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved cluster result to {filepath}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 80)
    print("GraphRAG Extraction Pipeline")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check API configuration
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("ERROR: OPENAI_API_KEY is not configured!")
        print("Please set your API key in .env file or environment variable.")
        print("Example: OPENAI_API_KEY=your-actual-api-key")
        sys.exit(1)
    
    if not MODEL_NAME or MODEL_NAME == "your-model-name-here":
        print("ERROR: MODEL_NAME is not configured!")
        print("Please set your MODEL_NAME in .env file or environment variable.")
        print("Example: MODEL_NAME=your-actual-model-name")
        sys.exit(1)
    
    if OPENAI_BASE_URL == "https://api.openai.com/v1":
        print("WARNING: Using default OpenAI endpoint. Remember to set OPENAI_BASE_URL for GLM-4.")
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    # Load data
    print("Loading data...")
    level1_nodes = load_level1_nodes(LEVEL1_FILE)
    print(f"  Level 1 nodes: {len(level1_nodes)}")
    
    level2_clusters = load_level2_clusters(LEVEL2_FILE)
    print(f"  Level 2 clusters: {len(level2_clusters)}")
    print()
    
    # Create directories
    create_directories(OUTPUT_DIR, DEBUG_DIR)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Debug directory: {DEBUG_DIR}")
    print()
    
    # Filter clusters if specific cluster ID requested
    if args.cluster_id:
        filtered_clusters = [c for c in level2_clusters if c.get("cluster_id") == args.cluster_id]
        if not filtered_clusters:
            print(f"ERROR: Cluster ID '{args.cluster_id}' not found")
            sys.exit(1)
        level2_clusters = filtered_clusters
        print(f"Processing only cluster: {args.cluster_id}")
        print()
    
    # Get already processed clusters if resuming
    processed_clusters = set()
    if args.resume:
        processed_clusters = get_processed_clusters(OUTPUT_DIR)
        if processed_clusters:
            print(f"Found {len(processed_clusters)} already processed clusters to skip")
            print()
    
    # Filter out processed clusters
    clusters_to_process = []
    for cluster in level2_clusters:
        cluster_id = cluster.get("cluster_id", "unknown")
        if cluster_id in processed_clusters:
            print(f"Skipping already processed cluster: {cluster_id}")
            continue
        clusters_to_process.append(cluster)
    
    if not clusters_to_process:
        print("No clusters to process. Exiting.")
        sys.exit(0)
    
    # Apply max clusters limit if specified
    if args.max_clusters:
        clusters_to_process = clusters_to_process[:args.max_clusters]
        print(f"Processing maximum {args.max_clusters} clusters")
        print()
    
    print(f"Clusters to process: {len(clusters_to_process)}")
    print()
    
    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No API calls will be made")
        print("=" * 80)
        for i, cluster in enumerate(clusters_to_process, 1):
            cluster_id = cluster.get("cluster_id", "unknown")
            cluster_size = cluster.get("cluster_size", 0)
            print(f"{i}. Cluster {cluster_id} ({cluster_size} Level 1 nodes)")
        print()
        print("Dry run complete. Exiting.")
        sys.exit(0)
    
    # Process clusters with progress bar
    print("Starting extraction...")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    
    with tqdm(total=len(clusters_to_process), desc="Processing clusters", unit="cluster") as pbar:
        for i, cluster in enumerate(clusters_to_process):
            cluster_id = cluster.get("cluster_id", f"unknown_{i}")
            cluster_size = cluster.get("cluster_size", 0)
            
            # Update progress bar description
            pbar.set_description(f"Cluster {cluster_id}")
            
            try:
                # Format cluster text with macro context and Level 1 nodes
                batch_text = process_cluster_batch(cluster, level1_nodes)
                
                # Save first batch for inspection if requested
                if args.inspect:
                    save_batch_text_for_inspection(batch_text, cluster_id, DEBUG_DIR)
                    print("\nInspection mode: saved first batch text and exiting")
                    if i == len(clusters_to_process) - 1:
                        sys.exit(0)
                    continue
                
                # Call LLM API
                result = call_llm_api(batch_text, client, timeout=args.timeout)
                
                # Save result
                save_cluster_result(result, cluster_id, OUTPUT_DIR)
                
                success_count += 1
                pbar.update(1)
                
                # Add delay if specified
                if args.delay > 0 and i < len(clusters_to_process) - 1:
                    time.sleep(args.delay)
                
            except Exception as e:
                fail_count += 1
                pbar.update(1)
                pbar.write(f"  ERROR: Cluster {cluster_id} failed: {e}")
                pbar.write("  Continuing to next cluster...")
                continue
    
    # Print summary
    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Started at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total clusters: {len(level2_clusters)}")
    print(f"Processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {len(processed_clusters)}")
    print(f"Results saved to: {OUTPUT_DIR}/cluster_*.json")
    print()
    
    if fail_count > 0:
        print("WARNING: Some clusters failed. Run with --resume to retry failed clusters.")
        sys.exit(1)
    else:
        print("SUCCESS: All clusters processed successfully!")
        print("Remember to merge the checkpoint files using merge_checkpoints.py when all clusters are complete.")


if __name__ == "__main__":
    main()