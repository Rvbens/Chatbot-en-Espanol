import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pickle, random, itertools, os, math, copy
import numpy as np

from pre_processing import Voc
from settings import *

import re
from pre_processing import process_punct, indexesFromSentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
save_dir = os.path.join("data", "Transformer_500k_UNK")
print(f'Loading: {save_dir}')

with open(save_dir + '/voc.pkl',  'rb') as f:
    voc   = pickle.load(f)
    
with open(save_dir + '/pairs.pkl','rb') as f:
    pairs = pickle.load(f)
    
with open("./data/loss_log.pkl", 'rb') as f:
    loss_log = pickle.load(f)

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


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len].clone().detach()
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask =  (torch.from_numpy(np_mask) == 0)
    return np_mask

def create_masks(input_seq, target_seq): 
    # creates mask with 0s wherever there is padding in the input
    input_msk = (input_seq != PAD_token).unsqueeze(1)

    # create mask as before
    target_pad = PAD_token
    target_msk = (target_seq != PAD_token).unsqueeze(1)
    size = target_seq.size(1) # get seq_len for matrix
    no_peak_mask = nopeak_mask(size)
    target_msk = target_msk & no_peak_mask
    return input_msk, target_msk

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

d_model = 512 #original 512
heads = 8
N = 6 #original 6
src_vocab = voc.num_words
trg_vocab = voc.num_words
model = Transformer(src_vocab, trg_vocab, d_model, N, heads,0.1)
model = model.to(device)


loadFilename = os.path.join(save_dir, 'checkpoints','transformer_checkpoint.tar')

if USE_CUDA:
    checkpoint = torch.load(loadFilename)
else:
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['params'])
voc.__dict__ = checkpoint['voc_dict']
model.eval()


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
    
    frst_dec_inp = torch.LongTensor([[SOS_token]]).cuda()
    trg_mask = nopeak_mask(1).cuda()
    
    out = model.out(model.decoder(frst_dec_inp, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1) #(bs,sl,voc_size)
    
    probs, ix = out[:, -1].data.topk(k) #(1,k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0) #(1,k)
   
    k_outs = torch.zeros(k, MAX_LENGTH).long().cuda()
    
    k_outs[:, 0] = SOS_token
    k_outs[:, 1] = ix[0] #first col with all the first k words
    
    e_outputs = torch.zeros(k, e_output.size(-2),e_output.size(-1)).cuda()
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
        trg_mask = nopeak_mask(i).cuda()
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
    input_batch = torch.LongTensor(indexes_batch).cuda() #(bs=1,seq_len)
    # Decode sentence with searcher
    tokens = searcher(model, input_batch)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

searcher = beam_search

def evaluateOneInput(input_sentence):
    input_sentence = process_punct(input_sentence.encode())
    # Evaluate sentence
    output_words = evaluate(model, searcher, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x =='SOS' or x == 'EOS' or x == 'PAD')]
    raw_ans = ' '.join(output_words)
    ans = reformatString(raw_ans)
    return ans

#bot cycle, receive input and outputs answer
def evaluateCycle():
    print("Enter q or quit to exit")
    input_sentence = ''
    while(1):
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        ans = evaluateOneInput(input_sentence)
        print('Bot:',ans)