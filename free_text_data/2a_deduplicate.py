import pandas as pd
from semhash import SemHash
from model2vec import StaticModel
import networkx as nx
from tqdm.auto import tqdm
import argparse
import os

def deduplicate_with_graph(input_csv: str, output_csv: str, threshold: float, model_name: str):
    """
    Performs semantic deduplication on a CSV file using a graph-based approach.

    Args:
        input_csv (str): Path to the input CSV file. Must contain a 'text' column.
        output_csv (str): Path to save the deduplicated CSV file.
        threshold (float): Similarity threshold (0 to 1) for connecting documents.
        model_name (str): Name of the pre-trained model to use for embeddings.
    """
    # 1. Load Model and Data
    print(f"Loading model: {model_name}...")
    m2v = StaticModel.from_pretrained(model_name)

    print(f"Reading data from: {input_csv}...")
    df = pd.read_csv(input_csv)
    df["id"] = df.index
    original_count = len(df)
    records = df.to_dict(orient="records")

    # 2. Compute Hashes and Get Baseline
    print("Computing semantic hashes for all documents...")
    semhash = SemHash.from_records(
        records=records,
        columns=["text"],
        model=m2v
    )

    # Calculate the result of a naive deduplication for comparison in the report
    naive_dedup_number = len(semhash.self_deduplicate(threshold=threshold).selected)

    # 3. Build Similarity Graph
    print(f"Building similarity graph with threshold: {threshold}...")
    G = nx.Graph()
    for rec in records:
        G.add_node(rec["id"], **rec)

    for rec, vec in tqdm(zip(records, semhash.index.vectors), total=original_count, desc="Finding similar pairs"):
        # Find neighbors above the similarity threshold
        nbrs = semhash.index.query_threshold(vec, threshold=threshold)[0]
        for nbr_rec, score in nbrs:
            u, v = rec["id"], nbr_rec["id"]
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, weight=score)

    # 4. Find Connected Components and Select Representatives
    print("Finding connected components in the graph...")
    components = list(nx.connected_components(G))

    representatives = []
    print("Selecting a representative from each component...")
    for comp in tqdm(components, desc="Processing components"):
        subg = G.subgraph(comp)
        # Select the node with the highest degree as the representative
        if len(subg) > 0:
            best_node = max(subg.degree(), key=lambda x: x[1])[0]
            representatives.append(best_node)

    # 5. Save Deduplicated Data
    df_reps = df[df["id"].isin(representatives)].reset_index(drop=True)
    df_reps = df_reps.drop(columns='id')
    graph_dedup_number = len(df_reps)
    
    print(f"\nSaving {graph_dedup_number} deduplicated rows to: {output_csv}")
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_reps.to_csv(output_csv, index=False)
    print("Deduplication complete.")

    # 6. Generate and Save Report
    removed_count = original_count - graph_dedup_number
    removed_percent = (removed_count / original_count) * 100 if original_count > 0 else 0
    report_path = os.path.splitext(output_csv)[0] + "_report.txt"

    report_content = (
        f"--- Semantic Deduplication Summary Report ---\n\n"
        f"Input File:              {input_csv}\n"
        f"Output File:             {output_csv}\n"
        f"Similarity Threshold:    {threshold}\n"
        f"Embedding Model:         {model_name}\n\n"
        f"--- Statistics ---\n"
        f"Original Documents:      {original_count}\n"
        f"Documents Kept:          {graph_dedup_number}\n"
        f"Documents Removed:       {removed_count} ({removed_percent:.2f}%)\n\n"
        f"--- Comparison ---\n"
        f"A naive deduplication would have kept {naive_dedup_number} documents.\n"
        f"The graph-based approach is more robust as it finds all connected items\n"
        f"and selects the best representative (most connected) from each cluster.\n"
    )

    print("\n" + report_content)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform graph-based semantic deduplication on a CSV file."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing a 'text' column."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the deduplicated output CSV file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for considering two documents as duplicates (default: 0.85)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="minishlab/potion-multilingual-128M",
        help="Name of the pre-trained sentence transformer model to use."
    )

    args = parser.parse_args()

    deduplicate_with_graph(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        threshold=args.threshold,
        model_name=args.model_name
    )