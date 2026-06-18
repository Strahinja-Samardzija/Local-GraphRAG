import json
import chromadb
from tqdm import tqdm
from stella_embeddings import StellaEmbeddings

def main():
    print("="*60)
    print(" [Contextualizer] Starting Context Application Pipeline ")
    print("="*60)
    
    print("\n[System] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Connect to your original base collection
    try:
        source_collection = client.get_collection("stella")
    except Exception as e:
        print(f"[ERROR] Base collection 'stella' not found: {e}")
        return

    # Create the new contextualized collection
    target_collection_name = "stella_contextual"
    print(f"[System] Getting or creating target collection: '{target_collection_name}'...")
    target_collection = client.get_or_create_collection(target_collection_name)

    # ---------------------------------------------------------
    # 1. Load the context mapping
    # ---------------------------------------------------------
    print("\n[Phase 1] Loading context mapping from final_contexts.json...")
    try:
        with open("final_contexts.json", "r", encoding="utf-8") as f:
            context_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] 'final_contexts.json' not found. Please ensure the file exists.")
        return
        
    context_map = {item["chunk_id"]: item["context_to_prepend"] for item in context_data}
    print(f" -> Found {len(context_map)} chunks slated for contextualization.")

    # ---------------------------------------------------------
    # 2. Fetch ALL data from the original collection
    # ---------------------------------------------------------
    print("\n[Phase 2] Fetching source data from ChromaDB...")
    print(" -> Pulling documents, metadata, and embeddings (this may take a moment)...")
    all_data = source_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    total_chunks = len(all_data["ids"])
    print(f" -> Successfully loaded {total_chunks} total chunks into memory.")

    # ---------------------------------------------------------
    # 3. Sort chunks into "Clone" vs "Re-embed" buckets
    # ---------------------------------------------------------
    clone_ids, clone_docs, clone_metas, clone_embs = [], [], [], []
    reembed_ids, reembed_docs, reembed_metas = [], [], []

    print("\n[Phase 3] Sorting chunks and injecting situational context...")
    
    # HCI Observable: Progress bar for the sorting loop
    for i in tqdm(range(total_chunks), desc="Processing Chunks", unit="chunk", smoothing=0.1):
        c_id = all_data["ids"][i]
        doc = all_data["documents"][i]
        meta = all_data["metadatas"][i] or {}
        emb = all_data["embeddings"][i]

        if c_id in context_map:
            # --- APPLY SITUATIONAL CONTEXT ---
            context_text = context_map[c_id]
            new_text = f"{context_text}\n\n{doc}"
            
            meta["has_context"] = True
            
            reembed_ids.append(c_id)
            reembed_docs.append(new_text)
            reembed_metas.append(meta)
        else:
            # --- DIRECT CLONE ---
            meta["has_context"] = False
            clone_ids.append(c_id)
            clone_docs.append(doc)
            clone_metas.append(meta)
            clone_embs.append(emb)

    print(f" -> Ready to clone: {len(clone_ids)} chunks")
    print(f" -> Ready to re-embed: {len(reembed_ids)} chunks")

    # ---------------------------------------------------------
    # 4. Execute database updates
    # ---------------------------------------------------------
    print("\n[Phase 4] Executing Database Writes...")
    batch_size = 5000 # ChromaDB/SQLite safe batch size

    # Step 4a: Instantly clone the unchanged chunks
    if clone_ids:
        print("\n -> Step A: Cloning unmodified chunks to new collection...")
        # HCI Observable: Progress bar for batched database writes
        for i in tqdm(range(0, len(clone_ids), batch_size), desc="Cloning to DB", unit="batch"):
            target_collection.add(
                ids=clone_ids[i:i+batch_size],
                documents=clone_docs[i:i+batch_size],
                metadatas=clone_metas[i:i+batch_size],
                embeddings=clone_embs[i:i+batch_size]
            )

    # Step 4b: Re-embed the altered chunks using your Stella wrapper
    if reembed_ids:
        print("\n -> Step B: Generating new embeddings for contextualized chunks...")
        embeddings_model = StellaEmbeddings()
        
        # Note: Your StellaEmbeddings wrapper already uses tqdm internally, 
        # so this will output a nice progress bar automatically.
        new_embeddings = embeddings_model.embed_documents(reembed_docs)

        print("\n -> Step C: Saving newly embedded chunks to ChromaDB...")
        # HCI Observable: Progress bar for batched database writes of new embeddings
        for i in tqdm(range(0, len(reembed_ids), batch_size), desc="Writing Contexts to DB", unit="batch"):
            target_collection.add(
                ids=reembed_ids[i:i+batch_size],
                documents=reembed_docs[i:i+batch_size],
                metadatas=reembed_metas[i:i+batch_size],
                embeddings=new_embeddings[i:i+batch_size]
            )

    print("\n" + "="*60)
    print(" [SUCCESS] Contextual Collection Built Successfully! ")
    print("="*60)
    print(f" Original Collection : 'stella' ({total_chunks} items, Untouched)")
    print(f" New Collection      : 'stella_contextual' ({len(clone_ids) + len(reembed_ids)} items)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()