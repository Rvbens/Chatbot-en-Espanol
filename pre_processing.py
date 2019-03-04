import os
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pickle

corpus_name = "OpenSubtitle"
save_dir = os.path.join("data", "save")
corpus = "./data/es.txt"

MAX_LENGTH = 20
LINES_USED = 500_000
MIN_COUNT  = 3    # Minimum word count threshold for trimming

def processLine(s):
    s = s.strip().lower().decode()
    s = re.sub(r"\.000",r" mil", s)
    s = re.sub(r"^-",       r"<GUION_INIC>", s)
    s = re.sub(r"-{2}",     r"<GUION_DOBL>", s)
    s = re.sub(r"\.{3}",    r"<TRIP_DOT>", s)
    s = re.sub(r"(\w)-(\w)",r"\1<GUION_INTER>\2", s)
    s = re.sub(r"([\).:!?])", r" \1", s) #separa puntuacion con espacio antes
    s = re.sub(r"([\(¡¿])", r"\1 ", s)   #separa puntuacion con espacio despues
    s = re.sub(r"([\"-,])", r" \1 ", s)  #espacio antes y despues
    #separa los tokens
    s = re.sub(r"<", r" <", s)
    s = re.sub(r">", r"> ", s)
    
    s = re.sub(r"\s<GUION_INTER>\s",r"-", s)
    return s

def loadLines(file, total_lines=10, max_lenght=MAX_LENGTH):
    with open(file, 'rb') as datafile:
        n=0
        for i,line in enumerate(datafile):
            if max(line.find(b"["),line.find(b"]")) == -1: #filtra los comentarios
                s = processLine(line)
                if len(s.split(' ')) < max_lenght: #filtra frases largas
                    n+=1
                    yield i,s
            if n==total_lines: break

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f"keep_words {len(keep_words)} / { len(self.word2index)} = "
              f"{len(keep_words) / len(self.word2index):.4f}")

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# Filter out sentences with trimmed words
def contain_trimmed(sentence, voc):
    for word in sentence.split(' '):
        if word not in voc.word2index:
            return True

def indexesFromSentence(sentence, voc):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def gen_pairs(how_many, voc, length):
    first_inpt = True
    n=0
    for out_idx,out_snt in loadLines(corpus,length):
        if first_inpt:
            inp_idx, inp_snt = out_idx,out_snt
            first_inpt = False
            continue

        if inp_idx+1 == out_idx: #frases contiguas
            if not contain_trimmed(inp_snt,voc) and not contain_trimmed(out_snt,voc):#every word is in Voc
                n+=1
                yield [indexesFromSentence(inp_snt, voc),indexesFromSentence(out_snt, voc)]
        if n==how_many:break
        inp_idx, inp_snt = out_idx,out_snt


def generate_data(lines=LINES_USED,length=MAX_LENGTH,trim=MIN_COUNT):
    voc = Voc(corpus_name)

    for _,s in loadLines(corpus,lines,length):
        voc.addSentence(s)

    voc.trim(trim)

    pairs = list(gen_pairs(lines, voc, length))
    
    #Guardar pairs y voc
    with open('data/pairs.pkl', 'wb') as file:
        pickle.dump(pairs, file)

    with open('data/voc.pkl', 'wb') as file:
        voc.__module__ = "Voc"
        pickle.dump(voc, file)

if __name__ == '__main__':
    generate_data()