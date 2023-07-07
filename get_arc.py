import numpy as np
import os
import sys
import pygraphviz as pgv


def process_cell(path, filename='cell_arc.png'):
    # Define operations
    operations = {
        0: 'none',
        1: 'max_pool_3x3',
        2: 'avg_pool_3x3',
        3: 'skip_connect',
        4: 'sep_conv_3x3',
        5: 'sep_conv_5x5',
        6: 'dil_conv_3x3',
        7: 'dil_conv_5x5',
    }

    # Define edges
    edges = {
        0: ['l -2', 'H1'],
        1: ['l -1', 'H1'],
        2: ['l -2', 'H2'],
        3: ['l -1', 'H2'],
        4: ['H1', 'H2'],
        5: ['l -2', 'H3'],
        6: ['l -1', 'H3'],
        7: ['H1', 'H3'],
        8: ['H2', 'H3'],
        9: ['l -2', 'H4'],
        10: ['l -1', 'H4'],
        11: ['H1', 'H4'],
        12: ['H2', 'H4'],
        13: ['H3', 'H4'],
        14: ['l -2', 'H5'],
        15: ['l -1', 'H5'],
        16: ['H1', 'H5'],
        17: ['H2', 'H5'],
        18: ['H3', 'H5'],
        19: ['H4', 'H5']
    }

    edge_colors = {
        'H1': 'red',
        'H2': 'blue',
        'H3': 'green',
        'H4': 'orange',
        'H5': 'purple',
    }

    # Load cell from npy file
    cell = np.load(path)

    # Initialize the final structure
    structure = {'l -2': [], 'l -1': [], 'H1': [],
                 'H2': [], 'H3': [], 'H4': [], 'H5': []}

    # Process the cell
    for edge, operation in cell:
        source, target = edges[edge]
        operation_name = operations[operation]
        structure[source].append(f'{operation_name} {target}')

    print(f'Cell path: {path}')
    print(f'Cell:\n {cell}')

    # Print the structure
    for source, operations in structure.items():
        print(f'{source}:')
        for operation in operations:
            print(f'    {operation},')
        print()

    # Draw the graph

    G = pgv.AGraph(directed=True, rankdir='LR')

    G.add_node('l -2', color='lightblue', style='filled', shape='box')
    G.add_node('l -1', color='lightblue', style='filled', shape='box')

    # Add nodes and edges to the graph
    for source, ops in structure.items():
        for operation in ops:
            op, target = operation.split()
            if op != 'none':
                G.add_edge(source, target, label=op, color=edge_colors[target])

    # Add concat node
    for node in structure.keys():
        if node != 'l -2' and node != 'l -1':
            G.add_edge(node, 'concat', color='black')

    # Set the layout to 'dot', which creates a hierarchical layout
    G.layout(prog='dot')

    # # Save the plot
    dir_path = os.path.dirname(path)

    # Draw the PyGraphviz graph
    G.draw(os.path.join(dir_path, filename))


# Use the function
if __name__ == "__main__":
    process_cell(sys.argv[1], sys.argv[2])
