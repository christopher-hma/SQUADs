"""Assortment of layers for use in models.py.

Author:
    Christopher Ma (christopherma922@gmail.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from util import masked_softmax

def mask_logits(target, mask):
    
    mask = mask.type(torch.float32)
    
    return target * mask + (1 - mask) * (-1e30)  


class PositionalEncoder(nn.Module):
    def __init__(self,embed_dim,max_seq_len = 80):
        
        super(PositionalEncoder,self).__init__()
        
        self.embed_dim = embed_dim
       
        self.pos_enc = torch.zeros(max_seq_len, embed_dim)
        
        for pos in range(max_seq_len):
            
            for i in range(0, embed_dim, 2):
                
                self.pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_dim)))
                
                self.pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_dim)))
                
        self.pos_enc = self.pos_enc.unsqueeze(0)
        
        self.pos_enc = nn.Parameter(self.pos_enc, requires_grad=False)

        
    def forward(self, x):
        
        seq_len = x.size(1)

        x = x + self.pos_enc[:,:seq_len]

        return x

class FeedForward(nn.Module):

      def __init__(self,input_dim,mid_dim,out_dim):

          super(FeedForward,self).__init__()

          self.input_dim = input_dim

          self.mid_dim = mid_dim

          self.out_dim = out_dim

          self.linear1 = nn.Linear(input_dim,mid_dim)


      def forward(self,input):
 
          hidden_in = self.linear1(input)

          output  = F.relu(hidden_in)   

          return output

class Depthwise_Separable_Conv(nn.Module):

      
      def __init__(self,in_channels,out_channels,kernel_size):

         
          super(Depthwise_Separable_Conv,self).__init__()
          
          self.in_channels = in_channels

          self.out_channels = out_channels

          self.kernel_size = kernel_size

          self.depth_conv = nn.Conv1d(in_channels = self.in_channels,out_channels = self.in_channels, kernel_size = self.kernel_size, stride=1, padding = self.kernel_size //2, groups = self.in_channels)
 
          self.point_conv = nn.Conv1d(in_channels = self.in_channels,out_channels = self.out_channels, kernel_size = 1, stride=1, padding=0)

        
      def forward(self,input):

          input = input.transpose(1,2)

          return F.relu(self.point_conv(self.depth_conv(input)).transpose(1,2))


class Output(nn.Module):

      def __init__(self,input_dim):

          super(Output,self).__init__()

          self.in_dim = input_dim * 2

          self.weight1 = nn.Linear(self.in_dim,1)

          self.weight2 = nn.Linear(self.in_dim,1)

      def forward(self,M0,M1,M2,mask):

           M_01 = torch.cat([M0,M1],dim = -1)

           M_02 = torch.cat([M0,M2],dim = -1)

           L_1 = self.weight1(M_01)
        
           L_2 = self.weight2(M_02)
        
           p1 = masked_softmax(L_1.squeeze(), mask, -1, True)
        
           p2 = masked_softmax(L_2.squeeze(), mask, -1, True)

           return p1,p2


class ContextQueryAtt(nn.Module):


      def __init__(self,input_dim,dropout = 0.1):

          super(ContextQueryAtt, self).__init__()

          self.input_dim = input_dim 

          self.Wq = torch.empty(input_dim,1)
  
          self.Wc = torch.empty(input_dim,1)

          self.Wcq = torch.empty(1,1,input_dim)

          nn.init.xavier_uniform_(self.Wq)
       
          nn.init.xavier_uniform_(self.Wc)
        
          nn.init.xavier_uniform_(self.Wcq)

          self.Wq = nn.Parameter(self.Wq)

          self.Wc = nn.Parameter(self.Wc)

          self.Wcq = nn.Parameter(self.Wcq)

          bias = torch.empty(1)
        
          nn.init.constant_(bias, 0)
        
          self.bias = nn.Parameter(bias)
        
          self.dropout = dropout
          

      def forward(self,context,query,c_mask,q_mask):

          bs = query.shape[0]

          q_len = query.shape[1]

          c_len = context.shape[1]
            
          c_mask = c_mask.view(bs,c_len, 1)
        
          q_mask = q_mask.view(bs, 1, q_len)

          context = F.dropout(context, p=self.dropout, training=self.training)
        
          query = F.dropout(query, p=self.dropout, training=self.training)

          query_sim = torch.matmul(query,self.Wq).transpose(1,2).expand([bs,c_len,q_len])

          context_sim = torch.matmul(context,self.Wc).expand([bs,c_len,q_len])

          context_query_sim = torch.matmul(context * self.Wcq, query.transpose(1,2))
          
          similarity = query_sim + context_sim + context_query_sim

          similarity += self.bias
            
          q_similarity = mask_logits(similarity,q_mask)
           
          c_similarity = mask_logits(similarity,c_mask) 

          S1 = F.softmax(q_similarity,dim = -1)

          S2 = F.softmax(c_similarity,dim = 1)

          A = torch.bmm(S1,query)

          B = torch.bmm(torch.bmm(S1,S2.transpose(1,2)),context)

          out = torch.cat([context, A, context * A, context * B], dim=2)

          return out



class SelfAttnLayer(nn.Module):


      def __init__(self,input_dim,num_heads,dropout = 0.1):

          super(SelfAttnLayer,self).__init__() 

          self.input_dim = input_dim

          self.num_heads = num_heads

          self.Wv = nn.Linear(self.input_dim,self.input_dim)

          self.Wk = nn.Linear(self.input_dim,self.input_dim)

          self.Wq = nn.Linear(self.input_dim,self.input_dim)

          self.dropout = nn.Dropout(dropout)

      def forward(self,input,mask):

          d = self.input_dim // self.num_heads

          V = self.Wv(input)

          K = self.Wk(input)

          Q = self.Wq(input)
        
          bs = input.shape[0]

          V = V.view(bs,-1,self.num_heads,d)

          K = K.view(bs,-1,self.num_heads,d)

          Q = Q.view(bs,-1,self.num_heads,d)

          Q = Q.transpose(1,2)

          K = K.transpose(1,2)

          V = V.transpose(1,2)


          weight = torch.matmul(Q,K.transpose(-2,-1))
        
          mask = mask.view(weight.shape[0],1,1,weight.shape[-1])
            
          weight = mask_logits(weight, mask)
        
  
          scaled_weight = weight * 1 /math.sqrt(d)

          scaled_weight = F.softmax(scaled_weight, dim = -1)

          scaled_weight = self.dropout(scaled_weight)

          v = torch.matmul(scaled_weight,V)

          output = v.transpose(1,2).contiguous().view(bs,-1,self.input_dim)

          return output
          
class Embedding(nn.Module):

  
      def __init__(self,input_dim,char_dim,word_dim):

          super(Embedding,self).__init__()

          self.input_dim = input_dim

          self.char_dim = char_dim

          self.word_dim = word_dim

          self.dropout = 0.1

          self.conv = nn.Conv2d(in_channels = char_dim, out_channels = input_dim,kernel_size = (1,5))

          self.proj = nn.Linear(input_dim + word_dim,input_dim)
          
          self.highway = Highway(2,input_dim)

      def forward(self,word_embed,char_embed):

          word_embed = F.dropout(word_embed, p=self.dropout, training=self.training)

          char_embed = char_embed.permute(0,3,1,2)

          char_embed = F.dropout(char_embed, p=self.dropout, training=self.training)
        
          char_embed = self.conv(char_embed)

          char_embed, _ = torch.max(char_embed,dim = -1)

          char_embed = char_embed.transpose(1,2)
 
          total_embed = torch.cat([word_embed,char_embed], dim = -1)

          total_embed = self.proj(total_embed)   

          output = self.highway(total_embed)

          return output           

          
class Highway(nn.Module):
    
    def __init__(self, num_layers, hidden_size):
        
        super(Highway, self).__init__()
        
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        
        for gate, transform in zip(self.gates, self.transforms):
            
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            
            t = F.relu(transform(x))
            
            x = g * t + (1 - g) * x

        return x        
             
          

class EncoderBlock(nn.Module):

    def __init__(self,input_dim,num_heads,repeat,k):

          super(EncoderBlock,self).__init__()
 
          self.input_dim = input_dim

          self.repeat = repeat

          self.convs = nn.ModuleList([Depthwise_Separable_Conv(input_dim,input_dim,k) for _ in range(self.repeat)])

          self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(self.repeat)])

          self.norm1 = nn.LayerNorm(input_dim)

          self.norm2 = nn.LayerNorm(input_dim)

          self.attn = SelfAttnLayer(input_dim,num_heads)

          self.feedforward = FeedForward(input_dim,input_dim,input_dim)

          self.dropout = 0.1

          self.PosEncoder = PositionalEncoder(input_dim,600)


    def forward(self,input,mask):

          input = self.PosEncoder(input)

          out = input

          for i in range(self.repeat):

              out = input

              input = self.norms[i](input)

              input = F.dropout(input, p = self.dropout, training=self.training)

              input = self.convs[i](input)

              input = input + out

          
          out = input

          input = self.norm1(input)

          input = F.dropout(input, p = self.dropout, training=self.training)

          input = self.attn(input,mask)

          input = input + out

          out = input

          input = self.norm2(input)

          input = F.dropout(input, p = self.dropout, training=self.training)
   
          input = self.feedforward(input)

          input = input + out

          return input
          
           

        
          


       

          

          
  
