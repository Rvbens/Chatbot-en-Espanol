import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

import pickle, random, itertools, os, math, re
import numpy as np

from settings import *
from transformer import Transformer, nopeak_mask 
from pre_processing import Voc, process_punct, indexesFromSentence

torch.set_grad_enabled(False)
USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data", "Transformer_500k_UNK")

with open(save_dir + '/voc.pkl',  'rb') as f:
    voc  = pickle.load(f)
    
#with open(save_dir + '/pairs.pkl','rb') as f:
    #pairs = pickle.load(f)
    

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(indexes_batch):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True) #orden por len of inp
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append([SOS_token]+pair[1])
    inp, lengths = inputVar(input_batch)
    output, mask, max_target_len = outputVar(output_batch)
    return inp, lengths, output, mask, max_target_len


def from_checkpoint(load_quant=False):
    print(f'Loading: {save_dir}')
    d_model = 512 #original 512
    heads = 8
    N = 6 #original 6
    src_vocab = voc.num_words
    trg_vocab = voc.num_words
    model = Transformer(src_vocab, trg_vocab, d_model, N, heads,0.1)
    model = model.to(device)

    if load_quant:
        loadFilename = os.path.join(save_dir, 'checkpoints','transformer_quant_checkpoint.tar')
        model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
    else:
        loadFilename = os.path.join(save_dir, 'checkpoints','transformer_checkpoint.tar')

    if USE_CUDA:
        checkpoint = torch.load(loadFilename)
    else:
        checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['params'])
    voc.__dict__ = checkpoint['voc_dict']
    model.eval()
    return model

def custom_capitalize(s):
    for i, c in enumerate(s):
        if c.isalpha():
            break
    return s[:i] + s[i:].capitalize()

def reformatString(l):
    s = l.strip().lower()
#     s = re.sub(r"<guion_inic>",r"", s)
    s = re.sub(r"\s+([.!?])", r"\1", s)
    s = re.sub(r"([¡¿])\s+", r"\1", s)
    s = re.sub(r"\s+", r" ", s)
    return custom_capitalize(s).strip()

def init_vars(src, model, k):
    src_mask = (src != PAD_token).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    frst_dec_inp = torch.LongTensor([[SOS_token]]).to(device)
    trg_mask = nopeak_mask(1).to(device)
    
    out = model.out(model.decoder(frst_dec_inp, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1) #(bs,sl,voc_size)
    
    probs, ix = out[:, -1].data.topk(k) #(1,k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0) #(1,k)
   
    k_outs = torch.zeros(k, MAX_LENGTH).long().to(device)
    
    k_outs[:, 0] = SOS_token
    k_outs[:, 1] = ix[0] #first col with all the first k words
    
    e_outputs = torch.zeros(k, e_output.size(-2),e_output.size(-1)).to(device)
    e_outputs[:, :] = e_output[0]
    
    return k_outs, e_outputs, log_scores

def k_best_outputs(k_outs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    k_outs[:, :i] = k_outs[row, :i]
    k_outs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return k_outs, log_scores

def beam_search(model, src, k=10):
    
    k_outs, e_outputs, log_scores = init_vars(src, model, k)
    src_mask = (src != PAD_token).unsqueeze(-2)
    ind = 0
    score= torch.tensor([[1 for i in range(k)]]).float()
    for i in range(2, MAX_LENGTH):
        trg_mask = nopeak_mask(i).to(device)
        out = model.out(model.decoder(k_outs[:,:i],e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        k_outs, log_scores = k_best_outputs(k_outs, out, log_scores, i, k)
        
        finish_outs = len(set((k_outs==EOS_token).nonzero()[:,0].cpu().numpy()))
        if finish_outs == k:
            break

    alpha = 0.7
    
    x = (k_outs==EOS_token).nonzero()
    EOS_idx = []
    out_idx=0
    for i in range(len(x)):
        if x[i][0] == out_idx:
            EOS_idx.append(i)
            out_idx+=1
    out_lens = x[EOS_idx][:,1]
    div = 1/(out_lens.type_as(log_scores)**alpha)
    score = log_scores * div
    _, ind = torch.max(score, 1)
    ind = ind.data[0]
    
    out = k_outs[random.choices([i for i in range(k)], torch.exp(score[0]))]
    length = (out[0]==EOS_token).nonzero()[0]
    out = out[0][1:length]

    return out

def evaluate(model, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    sentence = sentence.split()
    indexes_batch = [indexesFromSentence(sentence, voc)] #list of tokens
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).to(device) #(bs=1,seq_len)
    # Decode sentence with searcher
    tokens = searcher(model, input_batch)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

searcher = beam_search

def evaluateOneInput(input_sentence, model):
    input_sentence = process_punct(input_sentence.encode())
    # Evaluate sentence
    output_words = evaluate(model, searcher, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x =='SOS' or x == 'EOS' or x == 'PAD')]
    raw_ans = ' '.join(output_words)
    ans = reformatString(raw_ans)
    return ans

#bot cycle, receive input and outputs answer
def evaluateCycle(model):
    print("Enter q or quit to exit")
    input_sentence = ''
    while(1):
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        ans = evaluateOneInput(input_sentence,model)
        print('Bot:',ans)
