from keras.layers import Input, Embedding, LSTM, GRU, Dropout, add, RepeatVector, Dense
from preprocess import Data
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

data = Data()

story_maxlen = data.story_maxlen
vocab_size = data.vocab_size
query_maxlen = data.query_maxlen

inputs_train = data.inputs_train
queries_train = data.queries_train
answers_train = data.answers_train

inputs_test = data.inputs_test
queries_test = data.queries_test
answers_test = data.answers_test

RNN = GRU # LSTM could give different results

story = Input(shape=(story_maxlen,), name='story_inputs')
encoded_story = Embedding(vocab_size, 50, name='story_embedding')(story)
encoded_story = Dropout(0.3, name='story_dropout')(encoded_story)
# shape (samples, story_maxlen, 50)

question = Input(shape=(query_maxlen,), name='question_inputs')
encoded_question = Embedding(vocab_size, 50, name='question_embedding')(question)
encoded_question = Dropout(0.3, name='question_dropout')(encoded_question)
# shape (samples, query_maxlen, 50)
encoded_question = RNN(50, name='question_rnn')(encoded_question)
# shape (samples, 50)
encoded_question = RepeatVector(story_maxlen, name='question_repeat_vec')(encoded_question)
# shape (samples, story_maxlen, 50)

merged = add([encoded_story, encoded_question], name='story_question_add')
# shape (samples, story_maxlen, 50)
merged = RNN(50, name='story_question_rnn')(merged)
# shape (samples, 50)
merged = Dropout(0.3, name='story_question_dropout')(merged)

preds = Dense(vocab_size, activation='softmax', name='story_question_dense')(merged)
# shape (samples, vocab_size)

model = Model([story, question], preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# plot_model(model, 'model.png', show_shapes=True)

checkpointer = ModelCheckpoint('model_{epoch:02d}_{val_acc:.2f}.h5')
earlystopper = EarlyStopping(monitor='val_acc', verbose=2)
model.fit([inputs_train, queries_train], answers_train, 
          epochs=40, 
          validation_split=0.05,
          callbacks=[checkpointer])

loss, acc = model.evaluate([inputs_test, queries_test], answers_test)
print('Test loss, acc: {:.4f}, {:.4f}'.format(loss, acc))


