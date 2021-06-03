import networkx as nx
import nltk
import numpy as np
import pandas as pd
import spacy as sp
from nltk import ngrams
from pyvis.network import Network


class BiGramGraph:
    """
    A class used to transform a corpus given as a numpy array into a graph form of the
    2-gram representation.

    ...

    Attributes
    ----------
    Graph : nx.Graph
        The Graph Representation of The Ngram Input.
    N_nodes : int
        Number of Nodes in Graph.
    N_edges : int
        Number of Edges in Graph.
    In_Max_Deg : int
        Maximum In Degree in Graph.
    Out_Max_Deg : int
        Maximum Out Degree in Graph.
    In_Min_Deg : int
        Minimum In Degree in Graph.
    Out_Min_Deg : int
        Minimum Out Degree in Graph.

    Methods
    -------
    Example_Method(arg=None)
        Add info
    """

    def __init__(self, data):
        n = 2
        tokenized_text = ' '.join(data).split()
        ngram = ngrams(tokenized_text, n=n)
        ngram = list(ngram)

        n_frequencies = nltk.FreqDist(ngram)
        edges = list(dict(n_frequencies).keys())
        nodes = np.unique(np.array(edges).flatten())

        self.Graph = nx.DiGraph()
        self.Graph.add_nodes_from(nodes)
        for x, y in edges:
            self.Graph.add_edge(x, y, value=n_frequencies[(x, y)])

            # Graph Attributes
        self.N_nodes = len(nodes)
        self.N_edges = len(edges)
        self.In_Max_Deg = max(dict(self.Graph.in_degree).values())
        self.Out_Max_Deg = max(dict(self.Graph.out_degree).values())
        self.In_Min_Deg = min(dict(self.Graph.in_degree).values())
        self.Out_Min_Deg = min(dict(self.Graph.out_degree).values())
        self.Chromatic_N = nx.algorithms.coloring.greedy_color(self.Graph)

        # pos = nx.spiral_layout(self.Graph, resolution=3.5)
        # pos = nx.drawing.nx_pydot.graphviz_layout(self.Graph)

    def get_word_colors(self) -> pd.DataFrame:
        """
        :return: A pandas DataFrame, contains of each of the graph nodes with their color. Each of them colored using
            the Greedy Coloring algorithm.
        """
        return pd.DataFrame([self.Chromatic_N.values(),
                             self.Chromatic_N.keys()]).T.rename(columns={0: 'color', 1: 'word'})

    def Viz_Graph(self, notebook=False, height=500, width=900, directed=False):
        nt = Network(f'{height}px', f'{width}px', notebook=notebook, directed=directed)
        nt.set_options(
            'var options = { "physics": {"forceAtlas2Based": {"gravitationalConstant": -230,"springLength": 170,\
              "springConstant": 0,\
              "avoidOverlap": 1\
            },\
            "minVelocity": 0.75,\
            "solver": "forceAtlas2Based",\
            "timestep": 1\
          }\
        }\
        ')
        nt.from_nx(self.Graph)
        # nt.show_buttons(filter_=['physics'])
        nt.prep_notebook()
        return nt.show('nx.html')
