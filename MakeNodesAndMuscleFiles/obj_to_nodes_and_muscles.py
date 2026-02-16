import os
import sys

"""
This script converts a Wavefront OBJ mesh file to simulation Nodes and Muscles files.
It extracts unique vertices and edges from the OBJ file and writes them in the format
expected by the simulation. Usage:
    python obj_to_nodes_and_muscles.py <input.obj> [output_nodes] [output_muscles]
"""

def parse_obj(obj_path):
    """
    Reads an OBJ file and extracts:
      - Vertices (v lines)
      - Edges (from faces, f lines)
    Returns:
      - List of (x, y, z) tuples for each vertex
      - List of (a, b) tuples for each unique edge
    """
    vertices = []
    edges = set()
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse vertex coordinates
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith('f '):
                # Parse face indices and convert to edges
                parts = line.strip().split()[1:]
                idxs = [int(p.split('/')[0]) - 1 for p in parts]
                for i in range(len(idxs)):
                    a, b = idxs[i], idxs[(i+1)%len(idxs)]
                    edge = tuple(sorted((a, b)))
                    edges.add(edge)
    return vertices, sorted(edges)

def write_nodes(vertices, out_path):
    """
    Writes the Nodes file with header and node data:
      <count>\n<pulse>\n<up>\n<front>\n
      <id> <type> <x> <y> <z>\n
    All nodes are assigned type 0 by default.
    """
    with open(out_path, 'w') as f:
        f.write(f"{len(vertices)}\n0\n0\n1\n")
        for idx, (x, y, z) in enumerate(vertices):
            f.write(f"{idx} 0 {x:.6f} {y:.6f} {z:.6f}\n")

def write_muscles(edges, out_path):
    """
    Writes the Muscles file with header and edge data:
      <count>\n
      <id> <nodeA> <nodeB>\n
    """
    with open(out_path, 'w') as f:
        f.write(f"{len(edges)}\n")
        for idx, (a, b) in enumerate(edges):
            f.write(f"{idx} {a} {b}\n")

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python obj_to_nodes_and_muscles.py <input.obj> [output_nodes] [output_muscles]")
        sys.exit(1)

    obj_file = sys.argv[1]
    nodes_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(obj_file), "Nodes")
    muscles_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(obj_file), "Muscles")

    # Check OBJ file exists
    if not os.path.exists(obj_file):
        print(f"OBJ file '{obj_file}' not found.")
        sys.exit(1)

    # Parse OBJ and write output files
    vertices, edges = parse_obj(obj_file)
    write_nodes(vertices, nodes_file)
    write_muscles(edges, muscles_file)

    print(f"Wrote {len(vertices)} nodes to {nodes_file}")
    print(f"Wrote {len(edges)} muscles to {muscles_file}")

if __name__ == "__main__":
    main()
