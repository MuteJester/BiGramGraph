import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk import ngrams
from pyvis.network import Network
import utils

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

    def __init__(self, data, notebook=False):
        if not notebook:
            from tqdm import tqdm
            tqdm.pandas()
        else:
            from tqdm.notebook import tqdm
            tqdm.pandas()

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
        self._nlp = None
        self.Data = nx.algorithms.coloring.greedy_color(self.Graph)
        self.Data = pd.DataFrame([self.Data.values(),
                                  self.Data.keys()]).T.rename(columns={0: 'color', 1: 'word'})

        self.Edges = pd.DataFrame(edges, columns=['in', 'out'])
        self.Edges['weight'] = self.Edges.apply(lambda _z: n_frequencies[(_z['in'], _z['out'])], axis=1)

    def add_part_of_speech(self):
        import spacy as sp
        self._nlp = sp.load('en_core_web_sm')
        self.Data['pos'] = self.Data['word'].progress_apply(lambda _z: self._nlp(str(_z))[0].pos_)

    def get_Xi(self) -> int:
        """
        :return: The chromatic number of the graph.
        """
        return self.Data['color'].max() + 1

    def __repr__(self):
        n = self.N_nodes
        e = self.N_edges
        xi = self.get_Xi()
        return f'Number of words included: {n}\nNumber of edges included: {e}\nChromatic number: {xi}\n'

    def __getitem__(self, item) -> dict:
        return dict()

    def vectorize(self, string, method='chromatic'):
        if method == 'chromatic':
            tokens = string.split(' ')
            vec_form = []
            for tok in tokens:
                vec_form.append(self.Data.query(f'word == "{tok}"')['color'].values[0])
            return vec_form
        else:
            raise NameError('Bad Method')

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


G = BiGramGraph(utils.clean_trans_texts(utils.load_data(0, 'Lyrics')))
print(G.add_part_of_speech())