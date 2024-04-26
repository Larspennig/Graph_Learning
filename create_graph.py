import numpy as np
import networkx as nx

# Read the file
with open('conferenceroom_1.txt', 'r') as file:
    lines = file.readlines()

# Store the lines in a NumPy array
lines_array = np.array(lines)

# Create an empty graph
graph = nx.Graph()