import numpy as np


class TfIdfVectorizer():
    """
    Term Frequency - Inverce Document Frequency Vectorizer. 
    Allows for tokenized text with precomputed vocabulary as well as raw text

    Args:
    -----
    min_freq: float, default = 0.1
        Minimum number of documents required containing the term for it to be included in vocabulary
    
    max_freq: float, default = 0.5
        Maximum number of documents required containing the term for it to be included in vocabulary
    
    vocab: dict, default = None
        Vocabulary in the form of {str: int}
    
    stopwords: list, default = None
        Words to exclude from vocabulary
    """

    def __init__(self, min_freq: float=0.1, max_freq: float=0.5, vocab: dict=None, stopwords: list=None):
        self.__min_freq = min_freq
        self.__max_freq = max_freq
        self.vocab = vocab
        self.__df = None
        self.__stopwords = stopwords

    def fit(self, input: list):
        """
        Fit vectorizer to input. Computes vocabulary (if not passed) and Inverse Document Frequency for vocabulary terms.

        Args:
        -----
        input: list[str] or list[list[str]]
            Learning corpus
        """

        self.__n = len(input)
        if self.vocab:
            self._get_vocab(input, True)
        else:
            self._get_vocab(input)
        self.__idf = np.log(self.__n / np.array([self.__df[key] for key in self.vocab.keys()]))

    def transform(self, input: str) -> str:
        """
        Transform a given string to a vector.

        Args:
        --------
        input: str or list
            Sequence(s) to transform
        
        Returns:
        --------
        Numpy array of shape (N, T), where N is corpus size and T is vocabulary size
        """

        assert self.__df != None, 'Fit the vectorizer first! Document frequency for terms is unknown'
        assert self.__idf.shape[0] == len(self.vocab), 'Length of IDF vector must be equal to size of vocabulary'
        vocab_len = len(self.vocab.keys())
        
        if isinstance(input, list) and not isinstance(input[0], int) :
            output = np.zeros((self.__n, vocab_len))
            for i, doc in enumerate(input):
                if not isinstance(doc, list):
                    doc = doc.strip().split()
                for token in doc:
                    token = token.lower() if isinstance(token, str) else token
                    if not token in self.vocab.keys():
                        continue
                    j = self.vocab[token]
                    output[i, j] += 1
                output[i, :] = (output[i, :] / len(doc)) * self.__idf
        
        elif isinstance(input, str):
            output = np.zeros((1, vocab_len))
            doc = input
            if not isinstance(input, list):
                doc = doc.strip().split()
            for token in doc:
                token = token.lower() if isinstance(token, str) else token
                if not token in self.vocab.keys():
                    continue
                j = self.vocab[token]
                output[0, j] += 1
            output[0, :] = (output[0, :] / len(doc)) * self.__idf
        
        return output

    def fit_transform(self, input: list[str]) -> list[str]:
        """
        Fit the vectorizer to a learning corpus and transform the corpus.

        Args:
        --------
        input: list[str] or list[int]
            Learning corpus
        
        Returns:
        --------
        Numpy array of shape (N, T), where N is corpus size and T is vocabulary size
        """
        
        self.fit(input)
        output = self.transform(input)
        return output

    def _get_vocab(self, input: list[str], df_only: bool=False) -> list[str]:
        vocab = set()
        self.__df = {}

        for doc in input:
            if not isinstance(doc, list):
                doc = doc.strip().split()
            doc_df = {}
            for token in doc:
                if (not isinstance(token, int) and not token.isalpha()) or (self.__stopwords and (token in self.__stopwords)):
                    continue

                if not df_only:
                    vocab.add(token.lower())

                if doc_df.get(token, True):
                    doc_df[token] = False
                    self.__df[token] = self.__df.get(token, 0) + 1
        
        if not df_only:
            filtered_vocab = []
            for token in vocab:
                if self.__min_freq <= (self.__df[token] / self.__n) <= self.__max_freq:
                    filtered_vocab.append(token)
            self.vocab = dict(zip(filtered_vocab, range(len(filtered_vocab))))
