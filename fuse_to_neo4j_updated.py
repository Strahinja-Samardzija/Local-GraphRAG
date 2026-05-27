import os
import json
import hashlib
import networkx as nx
from tqdm import tqdm
from neo4j import GraphDatabase

JSON_DIR = "."
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"


def load_and_fuse_jsons(directory: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    print(f"Scanning directory: {directory}")

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("cluster_") and f.endswith(".json")
    ]
    print(f"Found {len(files)} cluster JSON files.")

    print("Pass 1/2: Adding all entities...")
    for filepath in tqdm(files, desc="Entities", unit="file"):
        cluster_tag = os.path.basename(filepath).replace(".json", "")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping {os.path.basename(filepath)}: {e}")
            continue

        for entity in data.get("entities", []):
            entity_id = str(entity.get("id"))
            if not entity_id:
                continue
            props = {k: v for k, v in entity.items() if k != "id"}
            G.add_node(entity_id, **props)

    print(f"  After Pass 1: {G.number_of_nodes()} nodes")

    print("Pass 2/2: Adding relationships and claims...")
    for filepath in tqdm(files, desc="Relationships", unit="file"):
        cluster_tag = os.path.basename(filepath).replace(".json", "")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        for rel in data.get("relationships", []):
            source = str(rel.get("source"))
            target = str(rel.get("target"))
            if not source or not target:
                continue
            if not G.has_node(source) or not G.has_node(target):
                print(f"  ⚠️  Relationship endpoint missing: {source} → {target}")
                if not G.has_node(source):
                    G.add_node(
                        source,
                        type="UNKNOWN",
                        description="Auto-created missing node from relationship",
                    )
                if not G.has_node(target):
                    G.add_node(
                        target,
                        type="UNKNOWN",
                        description="Auto-created missing node from relationship",
                    )
            props = {k: v for k, v in rel.items() if k not in ["source", "target"]}
            rel_type = props.pop("type", props.pop("label", "RELATED_TO"))
            G.add_edge(source, target, type=rel_type, **props)

        for claim in data.get("claims", []):
            subject = str(claim.get("subject"))
            obj = str(claim.get("object"))
            if not subject or not obj:
                print(f"  ⚠️  Skipping claim: missing subject/object")
                continue
            if not G.has_node(subject):
                print(f"  ⚠️  Auto-creating missing subject node: {subject}")
                G.add_node(
                    subject, type="UNKNOWN", description="Auto-created from claim"
                )
            if not G.has_node(obj):
                print(f"  ⚠️  Auto-creating missing object node: {obj}")
                G.add_node(obj, type="UNKNOWN", description="Auto-created from claim")

            claim_text = claim.get("claim", "")
            claim_hash = hashlib.md5(claim_text.encode()).hexdigest()[:12]
            claim_node_id = f"claim_{cluster_tag}_{claim_hash}"

            # FIX: Keep ALL claim properties including subject/object for direct querying
            claim_props = dict(claim)  # includes subject, object, claim, status
            claim_props["type"] = "Claim"
            claim_props["source_cluster"] = cluster_tag
            G.add_node(claim_node_id, **claim_props)

            G.add_edge(subject, claim_node_id, type="SUBJECT_OF")
            G.add_edge(claim_node_id, obj, type="ABOUT_OBJECT")

    print(f"Fusion Complete: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
    return G


def push_to_neo4j(G: nx.MultiDiGraph, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def setup_database(tx):
        print("Setting up constraints and indexes...")
        tx.run(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )
        tx.run(
            "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE"
        )
        tx.run(
            "CREATE FULLTEXT INDEX entity_description IF NOT EXISTS FOR (e:Entity) ON EACH [e.description]"
        )
        tx.run(
            "CREATE FULLTEXT INDEX claim_description IF NOT EXISTS FOR (c:Claim) ON EACH [c.description, c.claim]"
        )

    def write_graph(tx):
        for node_id, data in G.nodes(data=True):
            node_type = data.pop("type", "Unknown")
            base_label = "Claim" if node_type == "Claim" else "Entity"
            type_label = "".join(e for e in str(node_type) if e.isalnum())
            query = f"""
            MERGE (n:{base_label}:{type_label} {{id: $node_id}})
            SET n += $properties
            """
            tx.run(query, node_id=node_id, properties=data)

        for source, target, key, data in G.edges(keys=True, data=True):
            rel_type = data.pop("type", "RELATED_TO")
            rel_type = "".join(
                e for e in str(rel_type) if e.isalnum() or e == "_"
            ).upper()
            query = f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            """
            tx.run(query, source=source, target=target, properties=data)

    print("Connecting to Neo4j and writing data...")
    with driver.session() as session:
        session.execute_write(setup_database)
        print("Writing nodes and edges...")
        session.execute_write(write_graph)

    driver.close()
    print("Graph successfully pushed to Neo4j!")


if __name__ == "__main__":
    fused_graph = load_and_fuse_jsons(JSON_DIR)
    push_to_neo4j(fused_graph, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
