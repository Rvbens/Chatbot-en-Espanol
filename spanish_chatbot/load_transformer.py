import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

import pickle, random, itertools, os, math, re
import numpy as np

from .settings import *
from .transformer import Transformer, nopeak_mask 
from .pre_processing import Voc, process_punct, indexesFromSentence

torch.set_grad_enabled(False)

def model_device(model):
    return next(model.parameters()).device

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


def k_best_outputs(k_outs, out, log_scores, i, k):
    """ Gready searcher """
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    k_outs[:, :i] = k_outs[row, :i]
    k_outs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return k_outs, log_scores


class TransformerChatbot:
    def __init__(self,data_path=None,load_quant=True,use_cuda=False):
        """Load model with saved parameters

        Args:
        data_path: str. Path where the model is saved. Default: ./data/Transformer_500k_UNK
        load_quant: bool. Load the quantized version of the model.
        use_cuda: bool. Use of GPU.
        """
        data_path = os.path.join("data", "Transformer_500k_UNK") if data_path==None else data_path

        if not os.path.isdir(data_path) or len(os.listdir(data_path)) == 0:
            raise FileNotFoundError(f"No such file or directory: {data_path}, set the path to your model " 
                "directory or download the pre-trained one from https://github.com/Rvbens/Chatbot-en-Espanol "
                "and uncompress on ./data.")
            #download('transformer',load_quant)

        with open(data_path + '/voc.pkl',  'rb') as f:
            self.voc  = pickle.load(f)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.from_checkpoint(data_path,load_quant,use_cuda)
        self.searcher = self.beam_search

    def from_checkpoint(self,data_path,load_quant=False,use_cuda=torch.cuda.is_available()):
        print(f'Loading: {data_path}')
        d_model = 512 #original 512
        heads = 8
        N = 6 #original 6
        src_vocab = self.voc.num_words
        trg_vocab = self.voc.num_words
        model = Transformer(src_vocab, trg_vocab, d_model, N, heads,0.1)
        model = model.to(self.device)

        if load_quant and use_cuda:
            raise RuntimeError('Quantization not supported on CUDA backends')
        if load_quant:
            loadFilename = os.path.join(data_path, 'checkpoints','transformer_quant_checkpoint.tar')
            model = torch.quantization.quantize_dynamic(
                model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
            )
        else:
            loadFilename = os.path.join(data_path, 'checkpoints','transformer_checkpoint.tar')

        if use_cuda:
            checkpoint = torch.load(loadFilename)
        else:
            checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['params'])
        self.voc.__dict__ = checkpoint['voc_dict']
        model.eval()
        return model

    def init_vars(self, src, k):
        src_mask = (src != PAD_token).unsqueeze(-2)
        e_output = self.model.encoder(src, src_mask)
        
        frst_dec_inp = torch.LongTensor([[SOS_token]]).to(self.device)
        trg_mask = nopeak_mask(1).to(self.device)
        
        out = self.model.out(self.model.decoder(frst_dec_inp, e_output, src_mask, trg_mask))
        out = F.softmax(out, dim=-1) #(bs,sl,voc_size)
        
        probs, ix = out[:, -1].data.topk(k) #(1,k)
        log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0) #(1,k)
    
        k_outs = torch.zeros(k, MAX_LENGTH).long().to(self.device)
        
        k_outs[:, 0] = SOS_token
        k_outs[:, 1] = ix[0] #first col with all the first k words
        
        e_outputs = torch.zeros(k, e_output.size(-2),e_output.size(-1)).to(self.device)
        e_outputs[:, :] = e_output[0]
        
        return k_outs, e_outputs, log_scores

    def beam_search(self, model, src, k=10):
        k_outs, e_outputs, log_scores = self.init_vars(src, k)
        src_mask = (src != PAD_token).unsqueeze(-2)
        ind = 0
        score= torch.tensor([[1 for i in range(k)]]).float()
        
        for i in range(2, MAX_LENGTH):
            trg_mask = nopeak_mask(i).to(self.device)
            out = self.model.out(self.model.decoder(k_outs[:,:i],e_outputs, src_mask, trg_mask))
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

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        ### Format input sentence as a batch
        # words -> indexes
        sentence = sentence.split()
        indexes_batch = [indexesFromSentence(sentence, self.voc)] #list of tokens
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).to(self.device) #(bs=1,seq_len)
        # Decode sentence with searcher
        tokens = self.searcher(self.model, input_batch)
        # indexes -> words
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def evaluateOneInput(self, input_sentence):
        """ Give an answer to the input sentence using the model """
        input_sentence = process_punct(input_sentence.encode())
        # Evaluate sentence
        output_words = self.evaluate(input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x =='SOS' or x == 'EOS' or x == 'PAD')]
        raw_ans = ' '.join(output_words)
        ans = reformatString(raw_ans)
        return ans

    def evaluateCycle(self):
        """ Continous loop of inputs and answers"""
        print("Enter q or quit to exit")
        input_sentence = ''
        while(1):
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            ans = self.evaluateOneInput(input_sentence)
            print('Bot:', ans)
        