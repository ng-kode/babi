from __future__ import print_function
from functools import reduce
import re
import tarfile
import numpy as np
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import pickle

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

class Data():
    def __init__(self):
        try:
            path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                    '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                    '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise
        # Default QA1 with 1000 samples
        challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
        # QA1 with 10,000 samples
        # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
        with tarfile.open(path) as tar:
            train_stories = get_stories(tar.extractfile(challenge.format('train')))
            test_stories = get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        self.vocab_size = len(vocab) + 1
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        with open('word_idx.pkl', 'wb') as f:
            pickle.dump(self.word_idx, f)
        self.story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        self.query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        print('-')
        print('Vocab size:', self.vocab_size, 'unique words')
        print('Story max length:', self.story_maxlen, 'words')
        print('Query max length:', self.query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')
       
        self.inputs_train, self.queries_train, self.answers_train = self.vectorize_stories(train_stories)
        self.inputs_test, self.queries_test, self.answers_test = self.vectorize_stories(test_stories)

        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', self.inputs_train.shape)
        print('inputs_test shape:', self.inputs_test.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', self.queries_train.shape)
        print('queries_test shape:', self.queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', self.answers_train.shape)
        print('answers_test shape:', self.answers_test.shape)
        print('-')
        print("""Available data attributes:
                story_maxlen
                query_maxlen
                vocab_size
                inputs_train
                queries_train
                answers_train
                inputs_test
                queries_test
                answers_test""")
        print('-')
        print('preprocessing complete !')

    def vectorize_stories(self, data):
        """Return tokenized stories.
        Intake an array of tuples
        >>> vectorize_stories([(['Mary', 'moved', 'to', 'the', 'bathroom', '.', 'John', 'went', 'to', 'the', 'hallway', '.'], ['Where', 'is', 'Mary', '?'], 'bathroom')])
        ([story_seq], [query_seq], [answer_seq])
        """
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([self.word_idx[w] for w in story])
            queries.append([self.word_idx[w] for w in query])
            
            a_idx = self.word_idx[answer]            
            a_oht = np.zeros(len(self.word_idx) + 1)
            a_oht[a_idx] = 1.
            answers.append(a_oht)
        return (pad_sequences(inputs, maxlen=self.story_maxlen),
                pad_sequences(queries, maxlen=self.query_maxlen),
                np.array(answers))
    