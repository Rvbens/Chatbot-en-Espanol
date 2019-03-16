import os

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3

MAX_LENGTH = 20        # Tamaño de las frases
LINES_USED = 500_000 # Determina el tamaño de la parte de dataset que cogemos
MIN_COUNT  = 10         # Minimum word count threshold for trimming
WITH_UNK   = True      # Swap words not in voc for unk token, instead of deleting sentence

corpus = "./data/es.txt"
corpus_name = "OpenSubtitle_P3_500k_WT"
if WITH_UNK: corpus_name += "_UNK"
save_dir = os.path.join("data", corpus_name)ª