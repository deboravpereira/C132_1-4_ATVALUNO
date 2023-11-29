import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["estou me sentindo mal-humorado","eu não me senti humilhado"]

# Tokenização
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)

#Crie um dicionário chamado word_index
word_index = tokenizer.word_index

sequence = tokenizer.texts_to_sequences(sentence)
print(sequence[0:2])

# Preenchendo a sequência
padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
print(padded[0:2])

# Defina o modelo usando um arquivo .h5

# Teste o modelo

# Imprima o resultado


# {"alegria": 0, "medo": 1, "amor": 2, "tristeza": 3, "raiva": 4, "surpresa": 5}
