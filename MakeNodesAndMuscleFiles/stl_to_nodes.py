
#TODO: Improve documentation and comments in this file. Also, consider adding error handling where applicable.
import numpy as np
from stl import mesh
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python stl_to_nodes.py <input.stl> [output_nodes] [output_muscles]")
        sys.exit(1)

    stl_file = sys.argv[1]
    nodes_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(stl_file), "Nodes")
    muscles_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(stl_file), "Muscles")

    # Load STL and extract unique vertices
    m = mesh.Mesh.from_file(stl_file)
    vertices = m.vectors.reshape(-1, 3)
    unique_vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    ''' 
        Write Nodes file with header: <count><pulse>\n<up>\n<front>
        <count> = number of nodes
        <pulse> = pulse node (default = 0)
        <up> = top-most node (default = 0)
        <front> = 1 front-most node (default = 1)

    '''
    with open(nodes_file, "w") as f:
        f.write(f"{unique_vertices.shape[0]}\n0\n0\n1\n")
        for idx, (x, y, z) in enumerate(unique_vertices):
            f.write(f"{idx} 0 {x:.6f} {y:.6f} {z:.6f}\n")

    # Build muscles (edges) from triangle connectivity
    tris = inv.reshape(-1, 3)
    edges = np.vstack([
        tris[:, [0, 1]],
        tris[:, [1, 2]],
        tris[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)

    # Write Muscles file with header: <count>
    with open(muscles_file, "w") as f:
        f.write(f"{edges.shape[0]}\n")
        for idx, (a, b) in enumerate(edges):
            f.write(f"{idx} {a} {b}\n")

    print(f"Wrote {unique_vertices.shape[0]} nodes to {nodes_file}")
    print(f"Wrote {edges.shape[0]} muscles to {muscles_file}")

if __name__ == "__main__":
    main()
