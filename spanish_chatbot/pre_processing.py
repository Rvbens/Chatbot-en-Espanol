import os
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pickle
from .settings import *

def process_punct(s):
    s = s.strip().lower().decode()
    s = re.sub(r"\.000",r" mil", s)
    s = re.sub(r"^-",       r"<GUION_INIC>", s)
    s = re.sub(r"-{2}",     r"<GUION_DOBL>", s)
    s = re.sub(r"\.{3}",    r"<TRIP_DOT>", s)
    s = re.sub(r"{y:bi}",   r"<SPECIAL_1>", s)
    s = re.sub(r"(\w)-(\w)",r"\1<GUION_INTER>\2", s)
    s = re.sub(r"([\):!?])",r" \1", s)      #separa puntuacion con espacio antes
    s = re.sub(r"([\(¡¿])", r"\1 ", s)      #separa puntuacion con espacio despues
    s = re.sub(r"([\"-,¿\.}])", r" \1 ", s) #espacio antes y despues
    
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
                s = process_punct(line)
                s = s.strip().split()
                if len(s) < max_lenght: #filtra frases largas
                    n+=1
                    yield i,s
            if n==total_lines: break

# Default word tokens
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, spl_sent):
        for word in spl_sent:
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
        self.word2index = {"UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def has_trimmed_or_unk(spl_snt,voc):
    """ Filter out sentences with trimmed words or words not in the voc
    spl_snt is a list of words """
    for word in spl_snt:
        if word not in voc.word2index:
            return True

def indexesFromSentence(spl_snt, voc):
    idxs = []
    for word in spl_snt:
        try:
            idxs += [voc.word2index[word]]
        except:
            idxs += [UNK_token] #word2index["UNK"]=UNK_token
    return idxs + [EOS_token]

def gen_pairs(how_many, voc, length, with_unk):
    first_inp = True
    n=0
    for out_idx, out_snt in loadLines(corpus,length):
        
        if first_inp: #first cycle, set first input
            inp_idx, inp_snt = out_idx, out_snt
            first_inp = False
            continue

        if inp_idx+1 == out_idx: #frases contiguas (loadLines filter long sentences)
            ok_inp = True if with_unk else not has_trimmed_or_unk(inp_snt, voc) # if WITH_UNK==True, every inp is accepted
            ok_out = not has_trimmed_or_unk(out_snt, voc)
            
            if ok_inp and ok_out: #every word is in Voc
                n+=1
                yield [indexesFromSentence(inp_snt, voc),indexesFromSentence(out_snt, voc)]
        
        if n==how_many:break
        inp_idx, inp_snt = out_idx, out_snt #prepare next cycle


def generate_data(lines=LINES_USED,length=MAX_LENGTH,trim=MIN_COUNT, with_unk=WITH_UNK):
    voc = Voc(corpus_name)

    for _,s in loadLines(corpus,lines,length):
        voc.addSentence(s)

    voc.trim(trim)

    pairs = list(gen_pairs(lines, voc, length, with_unk))
    
    #Guardar pairs y voc
    with open('data/pairs.pkl', 'wb') as file:
        pickle.dump(pairs, file)

    with open('data/voc.pkl', 'wb') as file:
        #voc.__module__ = "Voc" #https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error
        pickle.dump(voc, file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a vocabulary and dataset of pair of sentences')
    parser.add_argument('--lines', metavar='path', default=LINES_USED, help='Number of lines from the orignial dataset to be processed')
    parser.add_argument('--max_len', metavar='path', default=MAX_LENGTH,  help='Max length of the sentence')
    parser.add_argument('--min_count', metavar='path', default=MIN_COUNT,  help='Min count of a word to be left of the vocabulary')
    parser.add_argument('--with_unk', metavar='path', default=WITH_UNK,  help='Change words not in the voc for an UNK_token, instead of removing the pair')
    args = parser.parse_args()

    generate_data(args.lines, args.max_len, args.min_count, args.with_unk)