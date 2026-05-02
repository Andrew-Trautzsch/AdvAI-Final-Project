"""
Generate graph data for the traffic analysis demo
Run this script to create graph_data.json
"""

import json
import numpy as np
from pathlib import Path

def generate_demo_graph_data():
    # Generate data
    graph_data = {
        "nodes": [],
        "edges": []
    }

    return graph_data

def save_graph_data(data, output_path="demo/graph_data.json"):
    # Save graph data to JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Graph data saved to {output_file}")

def load_from_notebook_results(notebook_path="traffic_analysis.ipynb"):
    # Load data from notebook
 
    return generate_demo_graph_data()

if __name__ == "__main__":
    print("Generating graph data for traffic analysis demo...")

    # generate graph data
    graph_data = generate_demo_graph_data()

    save_graph_data(graph_data)

    print("Demo graph data generated successfully!")
    print(f"Nodes: {len(graph_data['nodes'])}")
    print(f"Edges: {len(graph_data['edges'])}")
    print("\nOpen demo/index.html in your browser to view the interactive demo.")