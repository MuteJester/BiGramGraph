class ChromaticRandomWalker:
    """
   A class used to transform a corpus given as a numpy array into a graph form of the
   2-gram representation.

   ...

   Methods
   ----------
   generate : return a randomly generated sentence based on given arguments


   """


    def __init__(self, Graph):
        """

        Arguments
        ----------
        Graph : BiGramGraph the bigram graph based on which sentences will be generated

        """
        self.Graph = Graph
        self.max_xi = Graph.get_Xi()


    def __repr__(self):
        return self.Graph.__repr__()


    def generate_chromatic_vector(self, max_xi, size):
        chromatic_nums = list(range(max_xi))
        last_num = -1
        chrom_vec = []
        for i in range(size):
            index = np.floor((np.random.beta(1.5, 1.5, 1) * max_xi)[0])
            cur_choice = chromatic_nums[int(index)]
            while cur_choice == last_num:
                index = np.floor((np.random.beta(6, 2, 1) * max_xi)[0])
                cur_choice = chromatic_nums[int(index)]
            if cur_choice != last_num:
                last_num = cur_choice
                chrom_vec.append(cur_choice)
            else:
                continue
        self.Random_Chromatic_Vec = chrom_vec


    def calculate_path_weight(self, path):
        weight = 0
        start = path[0]
        for i in path[1:]:
            weight += self.Graph.Edges[(self.Graph.Edges['in'] == start) & (self.Graph.Edges.out == i)].weight.values[0]
            start = i
        return weight


    def generate(self, method='heaviest', vec_size=5, depth=10):
        """

        Arguments
        ----------
        method : str
            the protocl of path scoring via which the walker will choose its course available methods include:
            1) heaviest -> the max weighted path
            2) lightest -> the min weighted path
            3) density_max -> the max density path
            4) density_min -> the min density path
        vec_size : int
            the size of the randomly generated chromatic vector
        depth : int
            the maximum depth of search the walker will consider when choosing its next step


        """
        self.vec_size = vec_size
        self.generate_chromatic_vector(self.max_xi, self.vec_size)
        result = ' '
        first_word = self.Graph.Data[self.Graph.Data.color == self.Random_Chromatic_Vec[0]].sample(1).word.values[0]
        for n in tqdm(self.Random_Chromatic_Vec[1:]):
            # Calculate Best Path
            paths = []
            targets = self.Graph.Data[self.Graph.Data.color == n]
            targets = targets.sample(depth if len(targets) >= depth else len(targets))
            for target in tqdm(targets.word, leave=False):
                gen = self.Graph.get_Shortest_Simple_Path(first_word, target)
                paths.append(next(gen))
            weights = np.array([self.calculate_path_weight(i) for i in paths])
            if method == 'heaviest':
                best_walk = paths[np.argmax(weights)]
                first_word = targets.word.values[np.argmax(weights)]
            elif method == 'lightest':
                best_walk = paths[np.argmin(weights)]
                first_word = targets.word.values[np.argmin(weights)]
            elif method == 'density_max':
                weights = [calculate_path_density(self.Graph, nltk.ngrams(i, 2)) for i in paths]
                best_walk = paths[np.argmax(weights)]
                first_word = targets.word.values[np.argmax(weights)]
            elif method == 'density_min':
                weights = [calculate_path_density(self.Graph, nltk.ngrams(i, 2)) for i in paths]
                best_walk = paths[np.argmin(weights)]
                first_word = targets.word.values[np.argmin(weights)]
            del weights
            result += ' '.join(best_walk[:-1]) + ' '

        return result


def chromatic_distance(graph_1, graph_2):
        """
          Args
        ----------
        graph_1 : BiGramGraph
            the first graph to be compared against
        graph_2 : BiGramGraph
            the second graph to be compared against

        Returns the psi similarity coeffiecnt as presented in the paper
        return: int : psi similarity coeffiecnt
        """
        if 'pos' not in graph_1.Data.columns or 'pos' not in graph_2.Data.columns:
            raise PosError('Please Calculate PartofSpeech for Each Graph')

        overlaping_words = set(graph_1.Data['word'])
        overlaping_words = overlaping_words & set(graph_2.Data['word'])

        I = len(overlaping_words)

        chrom_ds = pd.DataFrame(index=list(overlaping_words))
        chrom_ds['chrom1'] = graph_1.Data.set_index('word').loc[overlaping_words].color
        chrom_ds['chrom2'] = graph_2.Data.set_index('word').loc[overlaping_words].color
        same_chrom_num = chrom_ds.apply(lambda x: np.mean(x) == x[0], axis=1)
        chrom_ds = chrom_ds[same_chrom_num].rename(columns={'chrom1': 'color'}).drop(columns=['chrom2'])

        # Epsilon
        E = 0
        chrom_ds['weight1'] = chrom_ds.index.to_series().apply(lambda x: graph_1.Graph.degree(x))
        # graph_1.Data.set_index('word').loc[overlaping_words].pos
        chrom_ds['weight2'] = chrom_ds.index.to_series().apply(lambda x: graph_2.Graph.degree(x))
        # same_weight = chrom_ds.apply(lambda x: np.max(x)<=2*np.min(x) ,axis=1)
        same_weight = chrom_ds[['weight1', 'weight2']].apply(lambda x: np.mean(x) == x[0], axis=1)
        same_weight = chrom_ds[same_weight]

        ICW = len(same_weight)
        IC = len(chrom_ds)

        return IC / I
