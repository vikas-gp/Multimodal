import pickle
import networkx as nx

print("===== LOADING FILE =====")
with open("skeleton_graphs.pkl", "rb") as f:
    data = pickle.load(f)

print("\n===== TYPE CHECK =====")
print("Type of loaded data:", type(data))

# -------------------------------------------------------
# If data is a LIST
# -------------------------------------------------------
if isinstance(data, list):
    print("\nData is a LIST")
    print("Length of list:", len(data))

    if len(data) > 0:
        G = data[0]
        print("Type of first element:", type(G))
    else:
        print("List is EMPTY")
        G = None

# -------------------------------------------------------
# If data is a DICT
# -------------------------------------------------------
elif isinstance(data, dict):
    print("\nData is a DICT")
    keys = list(data.keys())
    print("Number of keys:", len(keys))
    print("First 5 keys:", keys[:5])
    G = list(data.values())[0]
    print("Type of first value:", type(G))

# -------------------------------------------------------
# If it is a single graph
# -------------------------------------------------------
elif isinstance(data, nx.Graph):
    print("\nData itself is a NETWORKX GRAPH")
    G = data

else:
    print("\nUnknown format")
    G = None

# -------------------------------------------------------
# GRAPH INSPECTION
# -------------------------------------------------------
if G is not None and isinstance(G, nx.Graph):
    print("\n===== GRAPH INFORMATION =====")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # Print sample node attributes
    sample_nodes = list(G.nodes(data=True))[:5]
    print("\nSample 5 nodes with attributes:")
    print(sample_nodes)

    # Check if 'pos' exists
    has_pos = False
    if len(sample_nodes) > 0:
        has_pos = "pos" in sample_nodes[0][1]

    print("\nHas 'pos' attribute:", has_pos)

    # Print sample edges
    print("\nSample 5 edges:")
    print(list(G.edges())[:5])

else:
    print("\nERROR: No usable NetworkX graph found inside file!")
