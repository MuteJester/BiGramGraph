import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import networkx          as nx
import seaborn           as sns
from nltk import ngrams
from pyvis.network import Network
import os
import re
import nltk
import pydot
import spacy as sp
import pickle
from wordcloud import WordCloud
from tqdm.notebook import tqdm
from wordcloud import STOPWORDS
from string import punctuation
import enchant
import pickle
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline

wchecker = enchant.Dict("en_US")
nlps = sp.load('en_core_web_sm')


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
    Data : pandas DataFrame
        A Pandas DataFrame containing all words thier color label and POS and ENT tags if appropriate method is called prior.
    Edges : pandas DataFrame
        A Pandas DataFrame containing all edges and thier weights.    
    Name : str
        The name given to a graph (Usually to mark it with the name of its corresponding corpus).
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

    """

    def __init__(self, data=None, prebuild=None, notebook=False):
        """
        Attributes
        ----------
        data : list
            a list of texts to be converted into a BiGram Graph
        prebuild : list
            if prebuild is used than the bigram graph is constructed using the passed in list that sholud
            contain all elements that make up a bigram graph, mainly used in graph duplication and dumping as binary file.
        notebook : boolean
            if set to True then any resources loaded that are compatable with notebook instances will be adapted to notebook mode.

        Returns a BiGram-Graph representation of a given set of texts 
        return: A generator of cycles
        """
        if prebuild != None:
            self.Graph = prebuild[0]
            self.N_nodes = prebuild[1]
            self.N_edges = prebuild[2]
            self.In_Max_Deg = prebuild[3]
            self.Out_Max_Deg = prebuild[4]
            self.In_Min_Deg = prebuild[5]
            self.Out_Min_Deg = prebuild[6]
            self.Data = prebuild[7]
            self.Name = prebuild[8]
            self.Edges = prebuild[9]
        else:
            if not notebook:
                from tqdm import tqdm
                tqdm.pandas()
            else:
                from tqdm.notebook import tqdm
                tqdm.pandas()

            # merge text into a singal body and calculate bigram
            tokenized_text = ' '.join(data).split()
            ngram = ngrams(tokenized_text, n=2)
            ngram = list(ngram)

            # derive edge weights an unique words to be represented as nodes
            n_frequencies = nltk.FreqDist(ngram)
            edges = list(dict(n_frequencies).keys())
            nodes = np.unique(np.array(edges).flatten())

            # initiate an instance of a directed graph
            self.Graph = nx.DiGraph()
            self.Graph.add_nodes_from(nodes)
            # Set graph edges according to bigram pairs
            for x, y in edges:
                self.Graph.add_edge(x, y, value=n_frequencies[(x, y)])

            # ===================Graph Attributes ==============================
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
            self.Name = 'Default Name'

            self.Edges = pd.DataFrame(edges, columns=['in', 'out'])
            self.Edges['weight'] = self.Edges.apply(lambda _z: n_frequencies[(_z['in'], _z['out'])], axis=1)
            # =================================================================

    def add_part_of_speech(self):
        """
        Use spacy to extract part of speech tag for each node and append it to the "Data" attribute.
        """
        import spacy as sp
        self._nlp = sp.load('en_core_web_sm')
        self.Data['pos'] = self.Data['word'].progress_apply(lambda _z: self._nlp(str(_z))[0].pos_)

    def add_entities_of_speech(self):
        """
        Use spacy to extract part of speech tag for each node and append it to the "Data" attribute.
        """
        import spacy as sp
        self._nlp = sp.load('en_core_web_sm')

        def get_ent(_z):
            t = self._nlp(str(_z)).ents
            if len(t) > 0:
                return t[0].label_
            else:
                return 'NaN'

        self.Data['ent'] = self.Data['word'].progress_apply(get_ent)

    def get_Xi(self) -> int:
        """
        :return: The chromatic number of the graph.
        """
        return len(self.Data['color'].unique())

    def is_DAG(self):
        """
        Check if a BiGramGraph represent a directed acyclic graph
        return: Boolean: True if the graph is DAG else False
        """
        return nx.algorithms.dag.is_directed_acyclic_graph(self.Graph)

    def get_Diameter(self):
        """
        Returns the diameter of the graph
        return: Int: Diameter of the graph
        """
        return nx.algorithms.distance_measures.diameter(self.Graph)

    def get_Min_Edge_Cover(self):
        """
        Returns the minimum edge cover of the graph
        return: 
        """
        return nx.algorithms.covering.min_edge_cover(self.Graph)

    def get_Shortest_Simple_Path(self, start_node, end_node):
        """
          Attributes
        ----------
        start_node : str
            the node from which the path starts
        end_node : str
            the node at which the path ends

        Returns the shortest simple path between two words
        return: list: shortest path between two nodes
        """
        return nx.algorithms.simple_paths.shortest_simple_paths(self.Graph, source=start_node, target=end_node)

    def get_Eulerian(self):
        """

        Returns an euler graph if the given bigram garph is eulerian
        return: network x Graph: euler graph
        """
        if nx.is_eulerian(self.Graph):
            return nx.eulerian_circuit()(self.Graph)
        else:
            return 'Not Eulerian'

    def get_Volume(self, S):

        return nx.algorithms.cuts.volume(self.Graph, S)

    def get_Cycle(self, start_node):
        """
          Attributes
        ----------
        start_node : str
            the node from which the cycle starts


        Returns the cycle starting at a given word
        return: list: cycle starting at a given word
        """
        return nx.algorithms.cycles.find_cycle(self.Graph, start_node)

    def get_All_Unique_Cycles(self):
        """
        Returns all unique simple cycles contained inside the graph
        return: List of list where each inner list contains a sequence of nodes representing a cycle
        """
        hash_list = []
        unique_cycle = []
        for i in tqdm(range(self.N_nodes), leave=False):
            cyclye = self.get_Cycle(self.Data.word[i])
            c_hash = hash(str(self.get_Cycle(self.Data.word[i])))
            if c_hash not in hash_list:
                hash_list.append(c_hash)
                unique_cycle.append(cyclye)
        return unique_cycle

    def get_All_Simple_Cycles(self):
        """
        Returns a generator which output simple cycles 
        return: A generator of cycles
        """
        return nx.algorithms.cycles.simple_cycles(self.Graph)

    def get_Shortest_Path(self, source, target, weight=None, method='dijkstra'):
        """
          Attributes
        ----------
        source : str
            the node from which the path starts
        target : str
            the node at which the path ends   
        Returns the shortest path between two words
        return: list: path between two nodes
        """
        return nx.shortest_path(self.Graph, source=source, target=target, weight=weight, method=method)

    def is_Strongly_Connected(self):
        return nx.algorithms.components.is_strongly_connected(self.Graph)

    def get_Number_Strongly_Connected_Components(self):
        return nx.algorithms.components.number_strongly_connected_components(self.Graph)

    def get_Strongly_Connected_Components(self):
        return nx.algorithms.components.strongly_connected_components(self.Graph)

    def remove_self_loops(self):
        self.Graph.remove_edges_from(nx.selfloop_edges(self.Graph))

    def extract_K_Core(self, K=None, notebook=False):
        """
          Attributes
        ----------
        K : int
            the degree of weights in the K core extracted
        notebook : boolean
            the mode of the returend graph similar to the class constructor
        Returns the K core of the graph
        return: BiGramGraph: K core of graph
        """
        K_CORE = nx.algorithms.core.k_core(self.Graph, k=K)
        attributes = []
        # ===================Graph Attributes ==============================
        attributes.append(K_CORE)  # Graph
        attributes.append(K_CORE.number_of_nodes())
        attributes.append(K_CORE.number_of_edges())
        attributes.append(max(dict(K_CORE.in_degree).values()))
        attributes.append(max(dict(K_CORE.out_degree).values()))
        attributes.append(min(dict(K_CORE.in_degree).values()))
        attributes.append(min(dict(K_CORE.out_degree).values()))
        # Data = nx.algorithms.coloring.greedy_color(K_CORE)
        nodes = list(K_CORE.nodes())

        Data = self.Data.set_index('word').loc[nodes].reset_index()
        # Data['color'] = self.Data.set_index('word').loc[nodes].reset_index().color
        # Data = Data[['color','word']]
        attributes.append(Data)
        attributes.append(self.Name)

        weights = dict(K_CORE.edges)  # [('empty','street')]['value']

        Edges = pd.DataFrame(list(weights.keys()), columns=['in', 'out'])

        Edges['weight'] = Edges.apply(lambda _z: weights[(_z['in'], _z['out'])]['value'], axis=1)
        attributes.append(Edges)
        # =================================================================

        return BiGramGraph(prebuild=attributes, notebook=notebook)

    def __repr__(self):
        n = self.N_nodes
        e = self.N_edges
        xi = self.get_Xi()
        return f'Number of words included: {n}\nNumber of edges included: {e}\nChromatic number: {xi}\n'

    def __getitem__(self, item) -> dict:
        return dict()

    def vectorize(self, string, method='chromatic', seq_length=None, pad_with=None, strategy=None):
        """
         Attributes
       ----------
       string : str
           the target text to be vectroized
       method : str
           the method by which the text is vectorized , deafult is chromatic.
           methods include currently only chromatic vectorization
       seq_length : int
           the length of the vectorized output usually set to the maximum length string in a corpus
       pad_with : int
           the number with which the vectorizer will pad missing words only work when strategy argument is set to "pad_with"
       strategy : str
           which strategy sholud the vectorizer use when dealing with missing value currently only "pad_with" is supported

       Returns the vectorized version of the given string
       return: np.darray: vectorized text
       """

        if method == 'chromatic':
            if type(string) == str:
                if strategy is None:
                    vectorized = np.zeros(len(string))
                    for idx, word in enumerate(string.split(' ')):
                        query = self.Data.query(f'word == "{word}"').color.values
                        if len(query) == 0:
                            vectorized[idx] = float('nan')
                        else:
                            vectorized[idx] = query[0] + 1
                    return vectorized
                elif strategy == 'pad_with':
                    vectorized = (np.ones(max(len(string.split(' ')), seq_length)) * pad_with)
                    for idx, word in enumerate(string.split(' ')):
                        query = self.Data.query(f'word == "{word}"').color.values
                        if len(query) == 0:
                            vectorized[idx] = float('nan')
                        else:
                            vectorized[idx] = query[0] + 1
                    return vectorized
                else:
                    raise NotImplementedError('bad strategy')
            elif type(string) in [list, np.ndarray, pd.Series]:

                if strategy == 'pad_with':

                    vectorized = (np.ones(len(string), max(len(string), seq_length)) * pad_with)

                    for kdx, sentence in enumerate(string):
                        for idx, word in enumerate(sentence.split(' ')):
                            query = self.Data.query(f'word == "{word}"').color.values
                            if len(query) == 0:
                                vectorized[kdx, idx] = float('nan')
                            else:
                                vectorized[kdx, idx] = query[0] + 1
                    return vectorized
                else:
                    raise NotImplementedError('bad strategy')


        else:
            raise NameError('Bad Method')

    def dump(self):
        """
        Returns a list of all graph components which can be pickled and than reconstructen using the prebuild argument
        of the class constructor
        return: list: all elements that make up a bigram graph instance
        """
        return [self.Graph,
                self.N_nodes,
                self.N_edges,
                self.In_Max_Deg,
                self.Out_Max_Deg,
                self.In_Min_Deg,
                self.Out_Min_Deg,
                self.Data,
                self.Name,
                self.Edges
                ]

    def Viz_Graph(self, notebook=False, height=500, width=900, directed=False):
        """
        Returns an html file create via graphviz that represents the graph
        return: html: graph vizualization
        """
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
