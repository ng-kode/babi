from preprocess import Data
import re
from keras.models import load_model
import numpy as np

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

data = Data()

story = tokenize('Sandra went to the beach. John went to the hallway')
question = tokenize('Where is Sandra ?')
# this is just an dummy answer 
# which won't be used in our inference process
_ = 'bathroom'

story_vec, question_vec, _ = data.vectorize_stories([
  (story, question, _)
])

print('-')
print('loading model...')
model = load_model('model_25_1.00.h5')
print('-')
print('predicting...')
preds = model.predict([story_vec, question_vec])
pred_idx = np.argmax(preds[0])
reverse_word_idx = dict((v, k) for k, v in data.word_idx.items())
print('-')
print('result')
print('-')
print('story:')
print(' '.join(story))
print('question:')
print(' '.join(question))
print('answer:')
print(reverse_word_idx.get(pred_idx))