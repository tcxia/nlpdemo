import sys
import math
import numpy as np

class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])

        table_size = 1e8
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

class Ngram:
    def __init__(self, tokens) -> None:
        self.tokens = tokens
        self.count = 0
        self.score = 0.0

    def set_score(self, score):
        self.score = score
    
    def get_string(self):
        return "_".join(self.tokens)

class Corpus:
    def __init__(self, filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold, word_phrase_filename) -> None:
        i = 0
        file_pointer = open(filename, 'r')

        all_tokens = []
        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                token = token.lower()

                if len(token) > 1 and token.isalnum():
                    all_tokens.append(token)

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\r Reading corpus: %d" % i)

        sys.stdout.flush()
        print("\r Corpus read: %d" % i)

        file_pointer.close()

        self.tokens = all_tokens

        # for x in range(1, word_phrase_passes + 1):

    def build_ngrams(self, x, word_phrase_delta, word_phrase_threshold, word_phrase_filename):
        ngrams = []
        ngram_map = {}

        token_count_map = {}
        for token in self.tokens:
            if token not in token_count_map:
                token_count_map[token] = 1
            else:
                token_count_map[token] += 1
        i = 0
        ngram_l  = []
        for token in self.tokens:
            if len(ngram_l) == 2:
                ngram_l.pop(0)
            
            ngram_l.append(token)
            ngram_t = tuple(ngram_l)

            if ngram_t not in ngram_map:
                ngram_map[ngram_t] = len(ngrams)
                ngrams.append(Ngram(ngram_t))
        
            ngrams[ngram_map[ngram_t]].count += 1

            i+=1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rBuilding n-grams (%d pass): %d" % (x, i))
        sys.stdout.flush()
        print("\rn-grams(%d pass) build: %d" % (x, i))

        filtered_ngrams_map = {}
        file_pointer = open(word_phrase_filename + ("-%d" % x), 'w')

        i = 0
        for ngram in ngrams:
            product = 1
            for word_string in ngram.tokens:
                product *= token_count_map[word_string]
            ngram.set_score((float(ngram.count) - word_phrase_delta) / float(product))

            if ngram.score > word_phrase_threshold:
                filtered_ngrams_map[ngram.get_string()] = ngram
                file_pointer.write('%s %d\n' % (ngram.get_string(), ngram.count))

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rScoring n-grams: %d" % i)

        sys.stdout.flush()
        print("\rScored n-grams: %d, filtered n-grams: %d" % (i, len(filtered_ngrams_map)))
        file_pointer.close()

        # Combining the tokens
        all_tokens = []
        i = 0

        while i < len(self.tokens):

            if i + 1 < len(self.tokens):
                ngram_l = []
                ngram_l.append(self.tokens[i])
                ngram_l.append(self.tokens[i+1])
                ngram_string = '_'.join(ngram_l)

                if len(ngram_l) == 2 and (ngram_string in filtered_ngrams_map):
                    ngram = filtered_ngrams_map[ngram_string]
                    all_tokens.append(ngram.get_string())
                    i += 2
                else:
                    all_tokens.append(self.tokens[i])
                    i += 1
            else:
                all_tokens.append(self.tokens[i])
                i += 1

        print("Tokens combined")

        self.tokens = all_tokens




