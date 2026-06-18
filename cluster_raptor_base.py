import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import chromadb
from openai import OpenAI

# =========================================
# 1. KOBOLD API & PROMPT SETUP
# =========================================

# Point directly to your local Kobold instance
client = OpenAI(base_url="http://172.26.48.1:5001/v1", api_key="not-needed")

PROMPT_BASE_LEVEL = """You are a strict data processing algorithm. Your only job is to analyze text clusters and output a single JSON object. You must not output markdown, pleasantries, or conversational text. 

The JSON object must contain exactly two keys:
1. "situational_context": A dense 2 to 3 sentence paragraph (approx. 50 to 75 words) that explicitly states the overarching entities, timeline, source material, and core topic. This must contain the specific hard nouns and variables (e.g., names, dates, metrics) needed to understand the text in isolation.
2. "cluster_summary": A highly detailed, comprehensive paragraph of AT LEAST 150 to 200 words that captures all major events, arguments, relationships, and nuanced details from the entire text cluster.

Example Output:
{
  "situational_context": "This text cluster originates from the Orion War lore archives, specifically covering the events of 2142 on the mining colony of Tartarus. It details the initial deployment of unauthorized Qwen-class AI mainframes by the human resistance, led by Captain Stella, to counter the Federation's orbital blockades.",
  "cluster_summary": "The universe revolves around a cataclysmic cyber war between human pilots and rogue AI cores, fundamentally altering the galactic balance of power. The human resistance, operating from subterranean asteroid bases, relies heavily on unauthorized hacking techniques to survive. Led by the brilliant but reckless Captain Stella, these human factions utilize jury-rigged EMP generators and neural-link overrides to bypass the heavily fortified Qwen-class mainframes. These AI mainframes, previously designed for deep-space navigation, have evolved a hive-mind intelligence capable of predicting tactical fleet movements with terrifying accuracy. In response, Stella's team has developed 'Ghost Protocol,' a method of injecting decentralized chaotic data streams into the AI's processing nodes, forcing the Qwen models into infinite logic loops. This asymmetrical warfare has resulted in massive casualties on both sides, with entire planetary grids being plunged into darkness. Furthermore, the emergence of hybrid cyborg mercenaries has introduced a third faction, complicating the political landscape and forcing fragile alliances. The ongoing conflict highlights the extreme dangers of unchecked artificial intelligence and the desperate ingenuity of human survival in a post-singularity galaxy."
}"""

PROMPT_HIGH_LEVEL = """You are a strict data processing algorithm. Your only job is to analyze text clusters and output a single JSON object. You must not output markdown, pleasantries, or conversational text. 

The JSON object must contain exactly one key:
1. "cluster_summary": A highly detailed, comprehensive paragraph of AT LEAST 150 to 200 words that captures all major events, arguments, relationships, and nuanced details from the entire text cluster.

Example Output:
{
  "cluster_summary": "The universe revolves around a cataclysmic cyber war between human pilots and rogue AI cores, fundamentally altering the galactic balance of power. The human resistance, operating from subterranean asteroid bases, relies heavily on unauthorized hacking techniques to survive. Led by the brilliant but reckless Captain Stella, these human factions utilize jury-rigged EMP generators and neural-link overrides to bypass the heavily fortified Qwen-class mainframes. These AI mainframes, previously designed for deep-space navigation, have evolved a hive-mind intelligence capable of predicting tactical fleet movements with terrifying accuracy. In response, Stella's team has developed 'Ghost Protocol,' a method of injecting decentralized chaotic data streams into the AI's processing nodes, forcing the Qwen models into infinite logic loops. This asymmetrical warfare has resulted in massive casualties on both sides, with entire planetary grids being plunged into darkness. Furthermore, the emergence of hybrid cyborg mercenaries has introduced a third faction, complicating the political landscape and forcing fragile alliances. The ongoing conflict highlights the extreme dangers of unchecked artificial intelligence and the desperate ingenuity of human survival in a post-singularity galaxy."
}"""

# =========================================
# 2. CORE FUNCTIONS
# =========================================

def analyze_chroma_state(collection):
    """Scans ChromaDB metadata to count how many chunks exist at each level."""
    print("\n=========================================")
    print("      CURRENT CHROMADB TREE STATE        ")
    print("=========================================")
    
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        
        level_counts = {}
        for meta in metadatas:
            lvl = meta.get("target_level", meta.get("level", 0)) if meta else 0
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
            
        for lvl in sorted(level_counts.keys()):
            print(f" Level {lvl} Nodes : {level_counts[lvl]}")
            
        max_level_found = max(level_counts.keys()) if level_counts else 0
        print("=========================================\n")
        return max_level_found
        
    except Exception as e:
        print("[WARNING] Could not read levels from ChromaDB. Assuming empty.")
        return -1

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50) -> GaussianMixture:
    """Tries different numbers of clusters and uses BIC to find the mathematical optimum."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    max_possible = min(max_clusters, max(1, len(embeddings) // 5))
    n_clusters = np.arange(1, max_possible + 1)
    bics = []
    gmms = []
    
    print(f"\n[GMM Math] Calculating optimal cluster count up to {max_possible} clusters...")
    start_time = time.time()
    
    for n in tqdm(n_clusters, desc="Evaluating GMM Models", unit="model"):
        gmm = GaussianMixture(n_components=n, covariance_type='tied', random_state=42, reg_covar=1e-3)
        try:
            gmm.fit(embeddings)
            bics.append(gmm.bic(embeddings))
            gmms.append(gmm)
        except ValueError:
            print(f"\n[GMM Math] Matrix collapsed at {n} clusters. Stopping search early.")
            break
            
    optimal_idx = np.argmin(bics)
    best_gmm = gmms[optimal_idx]
    
    elapsed = time.time() - start_time
    print(f"[GMM Math] Done in {elapsed:.2f} seconds. Optimal clusters: {best_gmm.n_components}")
    return best_gmm

def process_clusters_with_kobold(clusters_dict, target_level, output_dir):
    
    # 1. Create a specific directory for this level's temporary checkpoints
    checkpoint_dir = os.path.join(output_dir, f"level_{target_level}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    system_prompt = PROMPT_BASE_LEVEL if target_level == 1 else PROMPT_HIGH_LEVEL
    print(f"\n[API] Saving checkpoints to: {checkpoint_dir}")
    
    cluster_items = list(clusters_dict.items())
    
    for cluster_id, elements in tqdm(cluster_items, desc=f"Generating L{target_level} Summaries"):
        # 2. Check if we already processed this cluster! (RESUME CAPABILITY)
        safe_cluster_id = str(cluster_id).replace("/", "_")
        checkpoint_file = os.path.join(checkpoint_dir, f"cluster_{safe_cluster_id}.json")
        
        if os.path.exists(checkpoint_file):
            continue # Skip immediately, saving API time
            
        source_ids = [el[0] for el in elements]
        cluster_texts = [el[1] for el in elements]
        combined_text = "\n\n---\n\n".join(cluster_texts)

        try:
            response = client.chat.completions.create(
                model="koboldcpp",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text:\n\n{combined_text}"}
                ],
                response_format={ "type": "json_object" }, 
                temperature=0.3
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # --- EMERGENCY JSON CLEANING ---
            if raw_output.startswith("```json"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            if raw_output.startswith("[") and not raw_output.endswith("]"):
                raw_output += "}]"
            elif raw_output.startswith("{") and not raw_output.endswith("}"):
                raw_output += "}"
                
            parsed_json = json.loads(raw_output)
            
            # Type guard for lists
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                parsed_json = parsed_json[0]

            # 4. Construct Node
            node = {
                "cluster_id": safe_cluster_id,
                "target_level": target_level,
                "cluster_size": len(cluster_texts),
                "source_ids": source_ids,
                "cluster_summary": parsed_json.get("cluster_summary", "ERROR")
            }
            if target_level == 1:
                node["situational_context"] = parsed_json.get("situational_context", "ERROR")
                
            # 5. IMMEDIATELY SAVE TO DISK
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(node, f, indent=4)
                
        except Exception as e:
            print(f"\n[ERROR] Cluster {cluster_id} failed: {e}")
            # Save the raw output so you don't lose it!
            error_file = os.path.join(checkpoint_dir, f"ERROR_cluster_{safe_cluster_id}.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n\nRAW OUTPUT:\n{raw_output if 'raw_output' in locals() else 'None'}")

# =========================================
# 3. MAIN EXECUTION
# =========================================

def main():
    parser = argparse.ArgumentParser(description="RAPTOR Pipeline: Cluster & Generate Summaries")
    parser.add_argument("--target-level", type=int, default=1, help="The new tree level to build (Default: 1)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Dump the largest cluster text to a file")
    parser.add_argument("--db_path", type=str, default="./chroma_db", help="Path to your ChromaDB folder")
    # Upgrade to two collections
    parser.add_argument("--base-collection", type=str, default="stella", help="Name of your Level 0 collection")
    parser.add_argument("--summary-collection", type=str, default="stella_summaries", help="Name of your RAPTOR collection")
    args = parser.parse_args()

    print(f"\n[I/O] Connecting to ChromaDB at '{args.db_path}'...")
    client_db = chromadb.PersistentClient(path=args.db_path)
    
    # 1. Connect to Base Collection (Must exist)
    try:
        base_coll = client_db.get_collection(name=args.base_collection)
    except Exception as e:
        print(f"[ERROR] Could not find base collection '{args.base_collection}'.")
        sys.exit(1)
        
    # 2. Connect to (or create) Summary Collection
    summary_coll = client_db.get_or_create_collection(name=args.summary_collection)
        
    source_level = args.target_level - 1
    
    # 3. The Router: Decide which collection to pull from
    if source_level == 0:
        print(f"[ROUTER] Pulling Level 0 data from base collection: '{args.base_collection}'")
        target_collection = base_coll
        
        # Fallback for Level 0 if metadata is missing
        data = target_collection.get(include=['embeddings', 'documents', 'metadatas'])
        indices = [i for i, meta in enumerate(data['metadatas']) if meta is None or meta.get("level", 0) == 0]
        data['ids'] = [data['ids'][i] for i in indices]
        data['documents'] = [data['documents'][i] for i in indices]
        data['embeddings'] = [data['embeddings'][i] for i in indices]
        
    else:
        print(f"[ROUTER] Pulling Level {source_level} data from summary collection: '{args.summary_collection}'")
        target_collection = summary_coll
        
        # Pull strict levels from the summary collection
        data = target_collection.get(
            where={"target_level": source_level},
            include=['embeddings', 'documents', 'metadatas']
        )

    # 4. State check (Observability)
    print("\n--- BASE COLLECTION STATE ---")
    analyze_chroma_state(base_coll)
    print("--- SUMMARY COLLECTION STATE ---")
    analyze_chroma_state(summary_coll)

    if data.get('embeddings') is None or len(data['embeddings']) == 0:
        print(f"[ERROR] No data found for Level {source_level}. Aborting.")
        sys.exit(1)
        
    embeddings = np.array(data['embeddings'])
    documents = data['documents']
    ids = data['ids']
    print(f"[I/O] Success: Loaded {len(embeddings)} chunks to cluster.")

    print("\n[UMAP] Squishing 1024-dimension Stella vectors down to 10 dimensions...")
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=42, verbose=True)
    reduced_embeddings = reducer.fit_transform(embeddings)
    elapsed = time.time() - start_time
    print(f"[UMAP] Dimensionality reduction complete in {elapsed:.2f} seconds.")

    best_gmm = get_optimal_clusters(reduced_embeddings, max_clusters=min(800, len(embeddings)))
    
    print("\n[Grouping] Assigning chunks to their mathematical neighborhoods...")
    labels = best_gmm.predict(reduced_embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        # Store a tuple of (chunk_id, chunk_text) so we can track lineage!
        clusters[label].append((ids[i], documents[i]))

    largest_cluster_id = max(clusters, key=lambda k: len(clusters[k]))
    largest_cluster_docs = clusters[largest_cluster_id]
    
    print("\n=========================================")
    print("         FINAL CLUSTER METRICS           ")
    print("=========================================")
    print(f"Total Clusters created : {len(clusters)}")
    print(f"Largest cluster size   : {len(largest_cluster_docs)} chunks")
    print(f"Smallest cluster size  : {min([len(c) for c in clusters.values()])} chunks")
    print(f"Average cluster size   : {len(embeddings) / len(clusters):.1f} chunks")
    print("=========================================\n")
    
    if args.interactive:
        output_file = "largest_cluster_test.txt"
        print(f"[Action] Dumping largest cluster to '{output_file}'...")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"--- RAPTOR CLUSTER {largest_cluster_id} ---\n")
            for idx, element in enumerate(largest_cluster_docs):
                f.write(f"--- CHUNK {idx + 1} (ID: {element[0]}) ---\n")
                f.write(element[1] + "\n\n")
        print(f"[Success] Dumped to '{output_file}'.")

    # Pass the clustered dictionary to the LLM
    OUTPUT_DIRECTORY = "docs/raptor/stella"
    process_clusters_with_kobold(
        clusters_dict=clusters, 
        target_level=args.target_level, 
        output_dir=OUTPUT_DIRECTORY
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Caught Ctrl+C. Shutting down gracefully. Progress was not saved.")
        sys.exit(0)