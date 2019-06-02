import os

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3

MAX_LENGTH = 40        # Tamaño de las frases
LINES_USED = 500_000 # Determina el tamaño de la parte de dataset que cogemos
MIN_COUNT  = 10         # Minimum word count threshold for trimming
WITH_UNK   = True      # Swap words not in voc for unk token, instead of deleting sentence
TIE_WEIGHTS= False

corpus = "./data/es.txt"
corpus_name = "Sentencepiece_Tfmr_500k"
if TIE_WEIGHTS: corpus_name += "_WT"
if WITH_UNK: corpus_name += "_UNK"
save_dir = os.path.join("data", corpus_name)
