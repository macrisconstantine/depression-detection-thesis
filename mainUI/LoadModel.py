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

# Assign variables and class labels
wordnet_lemmatizer = WordNetLemmatizer()
WINDOWS_SIZE = 10
labels = ["no", "mild", "moderate", "moderately severe", "severe"]
num_classes = len(labels)

# Ignore console warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Read the transcriptsall.csv file to a dataframe and assign column names and types
all_participants = pd.read_csv("mainUI/transcriptsall.csv", sep=",")
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

# Make a wordlist with and without stopwords from the answers in all interview transcript
all_participants_mix = all_participants.copy()
all_participants_mix["answer"] = all_participants_mix.apply(
    lambda row: text_to_wordlist(row.answer).split(), axis=1
)

all_participants_mix_stopwords = all_participants.copy()
all_participants_mix_stopwords["answer"] = all_participants_mix_stopwords.apply(
    lambda row: text_to_wordlist(row.answer, remove_stopwords=False).split(), axis=1
)

# Find the number of unique words in the wordlists
words = [w for w in all_participants_mix["answer"].tolist()]
words = set(itertools.chain(*words))
vocab_size = len(words)

words_stop = [w for w in all_participants_mix_stopwords["answer"].tolist()]
words_stop = set(itertools.chain(*words_stop))
vocab_size_stop = len(words_stop)

# Tokenize the answers in the wordlists
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

# Load the saved files with the previously trained LSTM models
model1 = load_model("mainUI/models/model_glove_lstm_b.h5")
model2 = load_model("mainUI/models/model_glove_2lstm_b.h5")

# Take input text and feed it to the given model to get a depression severity class prediction
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
    return (
        "\nThe provided text suggests "
        + labels[predicted_class]
        + " risk of depression."
    )
    
# The following functions are called in the LoadModel.py file to test the models 
def test_model1(text):
    return test_model(text, model1)

def test_model2(text):
    return test_model(text, model2)

# These following lines are commented out to avoid downloading the LLMs to the user's machine and to avoid any file reference errors 
# model3 = GPT4All("C:\\Users\\comac\\AppData\\Local\\nomic.ai\\GPT4All\\mistral-7b-instruct-v0.1.Q4_0.gguf", allow_download=False)
# model4 = GPT4All("C:\\Users\\comac\\AppData\\Local\\nomic.ai\\GPT4All\\wizardlm-13b-v1.2.Q4_0.gguf", allow_download=False)

# def test_llm(text, model):         
#     sysprompt = "use one word to classify depression severity of the input text: no, mild, moderate, moderately severe, or severe risk of depression"
#     with model.chat_session(sysprompt):
#         response1 = model.generate(text) 
#         return (
#         "\nThe provided text suggests "
#         + model.current_chat_session.pop()['content'].lower()
#         + " risk of depression."
#         )
        
# def test_model3(text):
#     return test_llm(text, model3)

# def test_model4(text):
#     return test_llm(text, model4)


# The following lines were for testing purposes only
# def menu(model):
#     sen = ""
#     while sen != "q":
#         sen = input("\nEnter a bit of text: ")
#         test_model(sen, model)


# menu(model2)

# sen = """
# I feel certain I am going mad again. I feel we can't go through another of those terrible times.
# And I shan't recover this time. I begin to hear voices, and I can't concentrate.
# So I am doing what seems the best thing to do. You have given me the greatest possible happiness. 
# You have been in every way all that anyone could be. I don't think two people could have been happier 
# till this terrible disease came. I can't fight any longer. I know that I am spoiling your life, that 
# without me you could work. And you will I know. You see I can't even write this properly. I can't read. 
# What I want to say is I owe all the happiness of my life to you. You have been entirely patient with me 
# and incredibly good. I want to say that - everybody knows it. If anybody could have saved me it would have been you. 
# Everything has gone from me but the certainty of your goodness. I can't go on spoiling your life any longer. 
# I don't think two people could have been happier than we have been.
# """

# test_model(sen, model2)
