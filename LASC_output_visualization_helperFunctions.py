"""
This script provides helper functions for visualizing the LASC output.

@author: zhang


"""
import os
os.chdir('D:/P/Pscript')

import numpy as np
from matplotlib import colors

def plot_behavior_raster(annotation_sequence, 
                         animal_num, 
                         axs,
                         start_frame,
                         stop_frame):
    '''
    plot behavioral rasters
    
    annotation_sequence (1d array of str): the predicted behavior annotations.
    animal_num (int): the animal number
    start_frame (int): the first frame showing in the plot
    stop_frame (int): the last frame showing in the plot
        
    '''
    stages = {
             'paralysis': 0, 
             'tonic seizure': 1, 
             'spasm': 2, 
             'clonic seizure': 3, 
             'recovery episode': 4,
             }
    class_to_number = {s: i for i, s in enumerate(stages)}
    
    
    annotation_num = []
    for item in annotation_sequence[start_frame:stop_frame]:
      annotation_num.append(class_to_number[item])
    
    cmap = colors.ListedColormap(['blue', 'red', 'orange', 'purple', 'green'])
    bounds=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    height = 200
    arr_to_plot = np.repeat(np.array(annotation_num)[:, np.newaxis].transpose(),
                                                    height, axis = 0)
    
    axs[animal_num].imshow(arr_to_plot, interpolation='none',cmap=cmap, norm=norm)
    axs[animal_num].set_yticks([])
    axs[animal_num].set_xlim(0,29000)


def get_state_transitions(annotation_sequence):
    '''
    detect change point for each animal
    
    Input: annotation_sequence (1d array of str): the predicted behavior annotations.
    
    Output: 
    state_transitions(list): the transitions for each session. e.g., [('paralysis', 'tonic seizure'), ('tonic seizure', 'spasm'), ('spasm', 'clonic seizure'), ('clonic seizure', 'spasm'), ('spasm', 'recovery episode'), ('recovery episode', 'clonic seizure'), ('clonic seizure', 'spasm'), ('spasm', 'clonic seizure'), ('clonic seizure', 'recovery episode')]
    trans_mt (5-by-5 array): 
    trans_mt_prob (5-by-5 array):
    
    '''
    stages = {
             'paralysis': 0, 
             'tonic seizure': 1, 
             'spasm': 2, 
             'clonic seizure': 3, 
             'recovery episode': 4,
             }
    class_to_number = {s: i for i, s in enumerate(stages)}
    
    annotation_num = []
    for item in annotation_sequence:
      annotation_num.append(class_to_number[item])
      
    state_transitions = []
    arr = np.array(annotation_num)
    diff_arr = np.diff(arr)
    change_index = np.where(diff_arr != 0)[0] # find indices
    state_1 = annotation_sequence[change_index]
    state_2 = annotation_sequence[change_index+1]
    state_transitions.append(list(zip(state_1, state_2)))

    trans_mt = np.zeros((5,5))
    for item in change_index:
        index1 = arr[item]
        index2 = arr[item+1]
        trans_mt[index1,index2] +=1
    
    trans_mt_prob = (trans_mt / trans_mt.sum(axis=1, keepdims=True))
   
    return state_transitions, trans_mt, trans_mt_prob


def plot_ethogram(
                  nodes,
                  node_size,
                  node_color,
                  node_XY,
                  edges,
                  ):
    '''
    Each node is a behavioral state. 
    the arrow indicate the state transitions. 
    the width of the arrow indicate the probability of this transition. 
    
    Parameters
    ----------
    nodes: list
    
    node_size : list, the same length of nodes. 
        the duration/friction of the stage in total seizing time.
    node_color : list
        DESCRIPTION.
    node_XY : dic
        e.g., {'A': (0, 0), 'B': (1, 1), 'C': (2, 0)}
    edges: list
        each element is like ('A', 'B', 0.8, 'blue'), from A to B, width 0.8 and blue
        
        
    Returns
    -------
    None.

    '''
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node, size, color in zip(nodes, node_size, node_color):
        G.add_node(node, size=size, color=color, pos=node_XY[node])
    
    # Add edges with probabilities and colors
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2], color=edge[3])
    
    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    node_sizes = [nx.get_node_attributes(G, 'size')[v] for v in G]
    node_colors = [nx.get_node_attributes(G, 'color')[v] for v in G]
    
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight']*20 for u, v in G.edges()]
    
    plt.figure(figsize=(4,3))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color=edge_colors, width=edge_widths, alpha=0.5)


    plt.axis('off')
    plt.show()


