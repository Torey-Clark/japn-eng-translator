import tensorflow as tf 

import fugashi # This is a Japanese Tokenizer since Japanese sentences do not use punctuation or spaces to seperate words.

import unicodedata
import re
import os
import time
import io

debug = False

jpn_tagger = fugashi.Tagger()

# Convert unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Convert all sentences into a standard format.
def preprocess_sentence_eng(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())

    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    sentence.rstrip().strip()

    sentence = '<start> ' + sentence + ' <end>'
    if debug:
        print(sentence)

    return sentence

# Use Fugashi to tokenize Japanese sentences.
def preprocess_sentence_jpn(sentence):
    
    sentence = unicode_to_ascii(sentence.lower().strip())

    words = [word.surface for word in jpn_tagger(sentence)]

    sentence = '<start> ' + sentence + ' <end>'
    if debug:
        print(sentence)

    return sentence

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, JAPANESE]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = []
    # For each line in the file, split it between English and Japanese
    for l in lines:
        temp_word_pairs = []
        words = l.split('\t')
        temp_word_pairs.append(preprocess_sentence_eng(words[0])) # Process English sentence
        temp_word_pairs.append(preprocess_sentence_jpn(words[1])) # Process Japanese sentence
        word_pairs.append(temp_word_pairs)

    print("Dataset Created")
    return zip(*word_pairs)

def tokenize(lang): 
    print("Tokenizing")
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    print("Tokenized")
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))

