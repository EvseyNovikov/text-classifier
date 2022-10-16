import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import re
import numpy as np
import os


script_dir = os.path.dirname(__file__)



mystem = Mystem()
russian_stopwords = stopwords.words("russian")

with open( os.path.join(script_dir, 'tokenizer.pickle'), 'rb') as handle:
    t = pickle.load(handle)

model = load_model( os.path.join(script_dir, 'tender.h5') )

def preprocess_text(text):
    text = re.sub(r"[^а-яА-Я\s\.\,]+", "", text)
    # print('text', text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    text = " ".join(tokens)
    return text


def predict(input_data):

    # Приводим текст к леммам, убираем цифры, знаки припинания, стоп-слова
    input_data = list(map(lambda x: preprocess_text(x), input_data))

    # Преобразуем текст в векторы
    sequences = t.texts_to_sequences(input_data)

    # Приводим векторы к одной длине
    data = pad_sequences(sequences, padding='post', maxlen=25)

    # Получаем предсказание из обученной модели
    result = model.predict(data)

    # На выходе берем значение самого высокого предсказания из всех
    output = list(map(lambda x:np.argmax(x), result))

    return output

print(predict(['Выполнение ремонтных работ фасада и наружной гидроизоляции индивидуального жилого дома по адресу: Московская область, Одинцовский район, р. п. Заречье, Кунцево-2 ']))

