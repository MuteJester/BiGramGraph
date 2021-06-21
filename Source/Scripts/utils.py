from wordcloud import STOPWORDS
from string import punctuation
from typing import Union, Iterable
import pandas as pd
import glob
import numpy as np

punctuation = punctuation + '“’‛‘”‹›»«'


def clean_trans_texts(text: Iterable) -> Union[pd.Series, pd.DataFrame]:
    """
    The following function cleans up a string (or a bunch of them) out of punctuations and stop words.
    :param text: A single/collection of strings.
    :return: A pandas Series/DataFrame (depends on the inputs type), that contains of the clean text(s).
    """

    def clean(_w: str):
        _new_w = ' '.join([word.lower() for word in _w.split() if
                           word not in ' '.join(punctuation).split() and word not in STOPWORDS])

        _new_w = ''.join(
            [''.join(char) for word in _new_w for char in word if char not in ' '.join(punctuation).split()])

        return _new_w

    if not isinstance(text, Iterable):
        raise Exception('The input is not iterable')

    if type(text) == pd.Series:
        _temp_text: pd.Series = text.dropna().apply(lambda _z: clean(_z))

    elif type(text) == str:
        _temp_text = pd.Series([text]).apply(lambda _z: clean(_z))

    elif type(text) == pd.DataFrame:
        _temp_text = text.dropna()
        for _c in text.columns:
            _temp_text[_c] = text[_c].apply(lambda _z: clean(_z))

    else:
        _temp_text = pd.Series(text).apply(lambda _z: clean(_z))

    return _temp_text


def load_data(file: Union[int, str], col=None) -> Union[pd.DataFrame, pd.Series]:
    """
    :param file: The name of the wanted file (not its path), or the file's index in an ordered list of all files in the
        data folder.
    :param col: Optional. The name of the wanted column(s) in the file.
    :return: pandas Series/DataFrame, contains of the file's data.
    """
    if type(file) == int:
        f_name = glob.glob('../../data/*')[file]
    elif type(file) == str:
        f_name = '../../data/' + file

    if col is None:
        return pd.read_csv(f_name)

    else:
        return pd.read_csv(f_name)[col]


def calculate_path_weight(Graph, path):
    weight = 0
    start = path[0]
    for i in path[1:]:
        weight += Graph.Edges[(Graph.Edges['in'] == start[0]) & (Graph.Edges.out == i[0])].weight.values[0]
        start = i
    return weight


def calculate_cycle_density(Graph, cycle):
    weight = 0
    for i in cycle:
        weight += np.sqrt(Graph.Graph.out_degree(i[0]) + Graph.Graph.in_degree(i[0]))
    return weight


def calculate_path_density(Graph, path):
    weight = 0
    for i in path:
        IN = Graph.Graph.out_degree(i[0])
        OUT = Graph.Graph.in_degree(i[0])
        if type(IN) != int:
            weight += np.sqrt(OUT)
        elif type(OUT) != int:
            weight += np.sqrt(IN)
        else:
            weight += np.sqrt(IN + OUT)
    return weight
