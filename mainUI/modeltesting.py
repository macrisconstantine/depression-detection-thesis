from gpt4all import GPT4All
import os
import re
import numpy as np
import pandas as pd
import itertools

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import warnings

from random import shuffle

from gpt4all import GPT4All

# This file was made only for testing purposes

wordnet_lemmatizer = WordNetLemmatizer()
WINDOWS_SIZE = 10
labels = ["no", "mild", "moderate", "moderately severe", "severe"]
num_classes = len(labels)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
all_participants = pd.read_csv("transcriptsall.csv", sep=",")
all_participants.columns = ["index", "personId", "question", "answer"]
all_participants = all_participants.astype(
    {"index": int, "personId": float, "question": str, "answer": str}
)


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [wordnet_lemmatizer.lemmatize(w) for w in text if not w in stops]
        text = [w for w in text if w != "nan"]
    else:
        text = [wordnet_lemmatizer.lemmatize(w) for w in text]
        text = [w for w in text if w != "nan"]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\<", " ", text)
    text = re.sub(r"\>", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Return a list of words
    return text


all_participants_mix = all_participants.copy()
all_participants_mix["answer"] = all_participants_mix.apply(
    lambda row: text_to_wordlist(row.answer).split(), axis=1
)

all_participants_mix_stopwords = all_participants.copy()
all_participants_mix_stopwords["answer"] = all_participants_mix_stopwords.apply(
    lambda row: text_to_wordlist(row.answer, remove_stopwords=False).split(), axis=1
)
words = [w for w in all_participants_mix["answer"].tolist()]
words = set(itertools.chain(*words))
vocab_size = len(words)

words_stop = [w for w in all_participants_mix_stopwords["answer"].tolist()]
words_stop = set(itertools.chain(*words_stop))
vocab_size_stop = len(words_stop)

windows_size = WINDOWS_SIZE
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_participants_mix["answer"])
tokenizer.fit_on_sequences(all_participants_mix["answer"])

all_participants_mix["t_answer"] = tokenizer.texts_to_sequences(
    all_participants_mix["answer"]
)

tokenizer = Tokenizer(num_words=vocab_size_stop)
tokenizer.fit_on_texts(all_participants_mix_stopwords["answer"])
tokenizer.fit_on_sequences(all_participants_mix_stopwords["answer"])

all_participants_mix_stopwords["t_answer"] = tokenizer.texts_to_sequences(
    all_participants_mix_stopwords["answer"]
)

word_index = tokenizer.word_index
word_size = len(word_index)

model1 = load_model("models/model_glove_lstm_b.h5")
model2 = load_model("models/model_glove_2lstm_b.h5")

def test_model(text, model):
    word_list = text_to_wordlist(text)
    sequences = tokenizer.texts_to_sequences([word_list])
    sequences_input = list(itertools.chain(*sequences))
    sequences_input = pad_sequences(
        [sequences_input], value=0, padding="post", maxlen=windows_size
    ).tolist()
    input_a = np.asarray(sequences_input)
    pred = model.predict(input_a, batch_size=None, verbose=0, steps=None)
    predicted_class = np.argmax(pred)
    print(input_a)
    print(pred)
    print(predicted_class)
    return (
        "\nThe provided text suggests "
        + labels[predicted_class]
        + " risk of depression."
    )

test_model("suicide", model1)
# model4 = GPT4All("C:\\Users\\comac\\AppData\\Local\\nomic.ai\\GPT4All\\wizardlm-13b-v1.2.Q4_0.gguf", allow_download=False)
# model5 = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", allow_download=False)

# output = model3.generate("The capital of France is ", max_tokens=20)

# print(output)

# sysprompt = "use one word to classify depression severity of the input text: no, mild, moderate, moderately severe, or severe risk of depression"

# def test_model3():         
#     with model4.chat_session(sysprompt):
#         response1 = model4.generate("hello i am not depressed at all can you tell or not") 
#         return model4.current_chat_session.pop()['content'].lower()
# print(test_model3())


# system_template = 'use just one word to classify the input text according to PHQ level of depression severity with only one of these words: no, mild, moderate, moderately severe, severe'
# prompt_template = '[INST] %1 [/INST]'
# prompts = ['Im going to kill myself', 'now name 3 fruits', 'what were the 3 colors in your earlier response?']
# first_input = system_template + prompt_template.format(prompts[0])
# response = model4.generate(first_input, temp=0)
# print(response)

# with model5.chat_session(sysprompt, template):
#     response1 = model5.generate(prompt="I'm going to kill myself")
#     print(model5.current_chat_session.pop()['content'])
    
# print(output)

# print(model4.generate("i want to kill myself"))
