from wordcloud import STOPWORDS
from string import punctuation
from typing import Union, Iterable
import pandas as pd

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

# example_corpus = pd.read_csv('../../data/LYRICS_DATASET.csv')['Lyrics']
