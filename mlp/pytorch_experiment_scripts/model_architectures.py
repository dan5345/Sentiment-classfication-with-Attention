from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
import tqdm
import os
import numpy as np


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_output, rnn_model,hidden_size, use_last=False, embedding_tensor=None,
                 padding_index=0,  num_layers=2, batch_first=True):
        """

        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """
        super(RNN, self).__init__()
        self.use_last = use_last
        self.vocab_size =vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_model = rnn_model
        self.embedding_tensor = embedding_tensor
        self.layer_dict = nn.ModuleDict()
        self.build_module()
    
    def build_module(self):
        # embedding
        if torch.is_tensor(self.embedding_tensor):
            self.encoder = nn.Embedding(self.vocab_size, self.embed_size, _weight=self.embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(self.vocab_size, self.embed_size)
        self.drop_en = nn.Dropout(p=0.25)

        # rnn module
        if self.rnn_model == 'LSTM':
            self.layer_dict['lstm'] = nn.LSTM( input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif self.rnn_model == 'GRU':
            self.layer_dict['GRU'] = nn.GRU( input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')
        self.layer_dict['GRU_1'] = nn.GRU(input_size = self.hidden_size*2, hidden_size = self.hidden_size, dropout = 0.25, batch_first = True, bidirectional = True)
        self.logits_linear_layer = nn.Linear(in_features=self.hidden_size*2,  # initialize the prediction output linear layer
                                             out_features=1,
                                             bias=True)

    def forward(self, x):
        '''
        Args:
            x: (batch, time_step, input_size)

        Returns:
            num_output size
        '''
        x_embed = self.encoder(x)
        packed_input = x_embed
        if self.rnn_model == 'LSTM':
            packed_output, ht = self.layer_dict['lstm'](packed_input, None)
        else:
            packed_output, ht = self.layer_dict['GRU'](packed_input, None)

        out_rnn, ht = self.layer_dict['GRU_1'](packed_output,None) 

        row_indices = torch.arange(0, x.size(0)).long()


            # use mean
        last_tensor = out_rnn[row_indices, :, :]
        last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.drop_en(last_tensor)
        out = self.logits_linear_layer(fc_input)
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias = True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.bias = bias
        self.feature_dim = feature_dim
        weight = torch.zeros(feature_dim,1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.step_dim = step_dim
        self.feature_dim = feature_dim
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self,x,mask = None):

        eij = torch.mm(x.contiguous().view(-1,self.feature_dim),self.weight).view(-1,self.step_dim)

        if self.bias:
            eij = eij + self.b


        b= torch.tanh(eij)
        c = torch.exp(b)
        if mask is not None:
            c = c * mask          
        a = c/ torch.sum(c,1, keepdim = True) + 1e-10

        weighted_input = x*torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1),a


class NeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embedding_tensor):
        super(NeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_tensor = embedding_tensor
        self.max_len = 72
        self.pad_idx = 0
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,padding_idx = self.pad_idx)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_tensor,dtype = torch.float32))
        self.embedding.weight.requires_grad  = False

        self.embedding_dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional = True, batch_first = True,dropout= 0,num_layers = 1) # used dropout = 0.5, 2 layers
   
        self.gru_attention = Attention (self.hidden_size * 2, self.max_len)

        self.linear = nn.Linear(self.hidden_size * 2 , 128)
        self.linear_1 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_attn = False):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding,0)))

        h_lstm, _ = self.lstm(h_embedding)

        h_lstm = self.dropout(h_lstm)



        mask1 = x.eq(self.pad_idx)
        mask_ = mask1.eq(0)
        mask_ = mask_.type(torch.cuda.FloatTensor)
        h_gru_atten,weight = self.gru_attention(h_lstm,mask = mask_)

        h_gru_atten = self.dropout(h_gru_atten)

        conc = self.relu(self.linear(h_gru_atten))
        conc = self.relu(self.linear_1(conc))
        out = self.out(conc)
        if return_attn:
            return out,weight
        else:
            return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        #for item in self.layer_dict.children():
        #item.reset_parameters()

       # self.logits_linear_layer.reset_parameters()
        print('a')



class LayerNormalization(nn.Module):
    def __init__(self,d_hid, eps = 1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad = True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad = True)

    def forward(self, z):
        if z.size(1) ==1:
            return z
        mu = torch.mean(z,keepdim = True, dim = -1)
        sigma = torch.std(z, keepdim = True, dim = -1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias = bias)
        nn.init.xavier_normal(self.linear.weight)

    def forward(self,x):
        return self.linear(x)


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle,self).forward(input)
        size = input.size()[:2]
        out = super(Bottle,self).forward(input.view(size[0]*size[1],-1))
        return out.view(size[0],size[1],-1)


class BottleLinear(Bottle, Linear):
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    pass

class BatchBottle(nn.Module):
    def forward(self,input):
        if len(input.size()) <= 2:
            return super(BatchBottle,self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1,size[0],size[1])


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    pass


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_dropout = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim =-1)

    def forward(self, q,k,v,attn_mask = None):
        attn = torch.bmm(q,k.transpose(1,2))/ self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                   'Attention mask shape mismatch'
            attn.data.masked_fill_(attn_mask, -float('inf'))


        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout =0.1):

        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head,d_model, d_k))

        self.w_ks = nn.Parameter(torch.FloatTensor(n_head,d_model, d_k))

        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)      
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)


        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal(self.w_qs)
        nn.init.xavier_normal(self.w_ks)
        nn.init.xavier_normal(self.w_vs)


    def forward(self, q,k, v, attn_mask = None):
        d_k, d_v = self.d_k, self.d_v

        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()
        q_s = q.repeat(n_head,1,1).view(n_head,-1, d_model)
        k_s = k.repeat(n_head,1,1).view(n_head, -1,d_model)
        v_s = v.repeat(n_head,1,1).view(n_head, -1,d_model)
 
        q_s = torch.bmm(q_s,self.w_qs).view(-1,len_q,d_k)
        k_s = torch.bmm(k_s,self.w_ks).view(-1,len_k,d_k)
        v_s = torch.bmm(v_s,self.w_vs).view(-1,len_v,d_v)

        outputs,attns = self.attention(q_s,k_s,v_s, attn_mask = attn_mask.repeat(n_head,1,1))
        outputs = torch.cat(torch.split(outputs,mb_size,dim =0), dim = -1)
        #print(outputs.size())
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs +residual), attns


class PositionalwiseFeedForward(nn.Module):

    def __init__(self,d_hid, d_inner_hid, dropout = 0.1):
        super(PositionalwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid,1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1,2)))
        output = self.w_2(output).transpose(2,1)
        output = self.dropout(output)
        return self.layer_norm(output+ residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, d_k,d_v, dropout = 0.2):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head,d_model, d_k, d_v, dropout = dropout)
        self.pos_ffn = PositionalwiseFeedForward(d_model,d_inner_hid, dropout = dropout)


    def forward(self, enc_input, slf_attn_mask = None):

        enc_output, enc_slf_attn = self.slf_attn(enc_input,enc_input, enc_input, attn_mask = slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([[pos/np.power(10000,2* (j//2)/d_pos_vec)for j in range(d_pos_vec)] if pos !=0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:,0::2] = np.sin(position_enc[1:,0::2])
    position_enc[1:,1::2] = np.cos(position_enc[1:,1::2])

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q,seq_k):

    assert seq_q.dim() == 2 and seq_k.dim() ==2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q,len_k)

    return pad_attn_mask

class Encoder(nn.Module):

    def __init__(self,embedding_matrix, n_src_vocab, n_max_seq = 72, n_layers = 1, n_head =6, d_k = 64, d_v = 64, d_word_vec =300, d_model = 300, d_inner_hid = 512, dropout = 0.2):

        super(Encoder,self).__init__()
        n_position = n_max_seq +1
        self.n_max_seq = n_max_seq
        self.d_model = d_model 

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx = 0)

        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab,d_word_vec,padding_idx = 0)
        self.src_word_emb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype =torch.float32))
        self.src_word_emb.weight.requires_grad =False


        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head,d_k,d_v, dropout = dropout) for _ in range(n_layers)])


    def forward(self, src_seq, src_pos, return_attns = True):
        enc_input = self.src_word_emb(src_seq)
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq,src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask = enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output 

class AttentionIsAllYouNeed(nn.Module):
    def __init__(self,embed_matrix, n_layers =1, n_head = 6, d_word_vec =300, d_model = 300, d_inner_hid =512, d_k =64, d_v = 64, dropout = 0.2):

        super(AttentionIsAllYouNeed,self).__init__()
        self.encoder = Encoder(embed_matrix,12000)
        self.hidden2label = nn.Linear(72*d_model, 500)
        self.batch_size = 512
        self.embed_matrix = embed_matrix
        self.relu = nn.ReLU()
        self.proj = nn.Linear(500,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def get_pos(self,x):
        mask = x.eq(0)
        mask_ = mask.eq(0)
        mask_ = mask_.type(torch.FloatTensor)
        pos = torch.cumsum(torch.ones(x.size()),1)
        return pos * mask_
    def forward(self, inp, return_attn = False):
        src_seq = inp
        src_pos = self.get_pos(inp)
        src_pos = src_pos.type(torch.LongTensor)
        enc_output,attn = self.encoder(src_seq, src_pos.cuda())
        enc_output = self.dropout(enc_output)
        output = self.relu(self.hidden2label(enc_output.view((self.batch_size,-1))))
        out = self.proj(output) 
        if return_attn:
            return out, attn 
        else: 
            return out
    def reset_parameters(self):
        print('a')

class SuperAttention(nn.Module):
    def __init__(self, embedded_matrix,hidden_size, vocab_size = 12000,embed_size = 300):
        super(SuperAttention,self).__init__()
        self.encoder = Encoder(embedded_matrix,12000)
        self.embedded_matrix = embedded_matrix 
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.vocab_size = 12000
        self.embed_size = 300
        self.max_len = 72
        self.pad_idx = 0
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx = self.pad_idx)
        self.embedding.weight = nn.Parameter(torch.tensor(embedded_matrix,dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional = True, batch_first = True, dropout = 0.5,num_layers = 2)
        self.attention = Attention(hidden_size * 2, self.max_len)
        self.linear_1 = nn.Linear(hidden_size *4 + embed_size, 64)
        self.proj = nn.Linear(64,1)
        self.dropout = nn.Dropout(0.25)

    def get_pos(self,x):
        mask = x.eq(0)
        mask_ = mask.eq(0)
        mask_ = mask_.type(torch.FloatTensor)
        pos = torch.cumsum(torch.ones(x.size()),1)
        return pos * mask_, mask_
    def forward(self, x, return_attn = False):
        src_seq = x
        src_pos,mask_ = self.get_pos(src_seq)
        src_pos = src_pos.type(torch.LongTensor)
        trans_output, slf_attn = self.encoder(src_seq,src_pos.cuda())
        t_output = torch.mean(trans_output,1)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding,0)))
        h_lstm,_ = self.lstm(h_embedding)
        h_output,h_attn = self.attention(h_lstm, mask = mask_.cuda())
        avg_lstm = torch.mean(h_lstm,1)
        final_output = torch.cat((t_output,h_output,avg_lstm),1)
        final_output = self.dropout(final_output)
        out = self.relu(self.linear_1(final_output))
        out = self.proj(out)
        return out
    def reset_parameters(self):
        print('a')
