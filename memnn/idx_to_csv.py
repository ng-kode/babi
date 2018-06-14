import pickle
import pandas as pd

word_idx = None
with open('word_idx.pkl', 'rb') as f:
  word_idx = pickle.load(f)

reverse_word_idx = dict((v, k) for k, v in word_idx.items())

df = pd.DataFrame.from_dict(reverse_word_idx, orient='index')
print(df.head())
df.to_csv('word_idx.csv', header=False, mode='w')
