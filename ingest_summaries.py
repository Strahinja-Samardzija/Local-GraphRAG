import json
import argparse
import chromadb
from tqdm import tqdm
from stella_embeddings import StellaEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Ingest RAPTOR JSON summaries into ChromaDB.")
    parser.add_argument("--json", required=True, help="Path to the JSON file containing summaries")
    # Changed default to a unified collection for all summary levels
    parser.add_argument("--collection", default="stella_summaries", help="Target ChromaDB collection name")
    args = parser.parse_args()

    print("="*60)
    print(" [RAPTOR] Starting Multi-Level Summary Ingestion ")
    print("="*60)

    # 1. Load JSON Data
    print(f"\n[1/3] Loading summaries from '{args.json}'...")
    try:
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read JSON: {e}")
        return
    
    if not data:
        print("[ERROR] JSON file is empty.")
        return

    # HCI Observable: Detect which levels are in this file
    levels_found = set(item.get('target_level', 'unknown') for item in data)
    print(f" -> Successfully loaded {len(data)} summary nodes.")
    print(f" -> Detected target levels in this file: {list(levels_found)}")

    # 2. Format for ChromaDB
    print("\n[2/3] Formatting metadata and preparing batches...")
    ids, docs, metas = [], [], []
    
    for item in data:
        # Safely get properties
        t_level = item.get("target_level", 0)
        c_id = item.get("cluster_id", "unknown")
        
        # Create a unique ID for the summary node that inherently tracks its level
        node_id = f"level_{t_level}_cluster_{c_id}"
        
        # ChromaDB requires metadata values to be strings, ints, or floats.
        source_ids_str = ",".join(item.get("source_ids", []))
        
        ids.append(node_id)
        docs.append(item.get("cluster_summary", ""))
        metas.append({
            "cluster_id": str(c_id),
            "target_level": t_level,
            "cluster_size": item.get("cluster_size", 0),
            "source_ids": source_ids_str,
            "node_type": "summary"
        })

    # 3. Embed and Save
    print(f"\n[3/3] Embedding summaries and saving to database collection '{args.collection}'...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(args.collection)
    
    # Initialize Stella model
    print(" -> Initializing embedding model...")
    embeddings_model = StellaEmbeddings()
    
    print("\n -> Generating vector embeddings for summaries...")
    embeddings = embeddings_model.embed_documents(docs)

    # Upsert to ChromaDB in chunks
    batch_size = 5000
    print("\n -> Upserting into ChromaDB...")
    for i in tqdm(range(0, len(ids), batch_size), desc="Writing to DB", unit="batch"):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size]
        )

    print("\n" + "="*60)
    print(" [SUCCESS] Summaries Ingested Successfully! ")
    print("="*60)
    print(f" Target Collection: '{args.collection}'")
    print(f" Levels Processed : {list(levels_found)}")
    print(f" Total Nodes Added: {len(ids)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()