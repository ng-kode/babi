from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from preprocess import Data

data = Data()
story_maxlen = data.story_maxlen
query_maxlen = data.query_maxlen
vocab_size = data.vocab_size
inputs_train = data.inputs_train
queries_train = data.queries_train
answers_train = data.answers_train
inputs_test = data.inputs_test
queries_test = data.queries_test
answers_test = data.answers_test

# placeholders
input_sequence = Input((story_maxlen,), name='story_inputs')
question = Input((query_maxlen,), name='question_inputs')

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential(name='story_m_embed_dropout')
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential(name='story_c_embed_dropout')
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential(name='question_embed_dropout')
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2), name='story_m_question_dot')
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c], name='match_story_c_add')  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1), name='match_permute')(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded], name='match_question_concat')

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = LSTM(32, name='answer_lstm')(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = Dropout(0.3, name='answer_dropout')(answer)
answer = Dense(vocab_size, name='answer_dense')(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)

# compile
print('-')
print('Compiling...')
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# create model.png
# plot_model(model, 'model.png', show_shapes=True)

# train
checkpointer = ModelCheckpoint('model_{epoch:02d}_{val_acc:.2f}.h5')
model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          epochs=120,
          validation_data=([inputs_test, queries_test], answers_test),
          callbacks=[checkpointer])
