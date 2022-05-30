"""Top-level model classes.

Author:
    Christopher Ma email: christopherma922@gmail.com
"""

import layers
import torch
import torch.nn as nn

import torch.nn.functional as F

class QANet(nn.Module):
    
  
    def __init__(self, word_vectors, char_vectors, hidden_size, num_heads,drop_prob=0.):
        
        super(QANet, self).__init__()

        self.dropout = drop_prob

        self.char_emb = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)

        
        self.emb = layers.Embedding(
                                      input_dim = hidden_size,
                                      char_dim = char_vectors.shape[1],
                                      word_dim = word_vectors.shape[1]
                                   )

          

      
        self.enc = layers.EncoderBlock(input_dim = hidden_size,num_heads = num_heads,repeat = 4,k = 7)



      
        self.att = layers.ContextQueryAtt(input_dim = hidden_size)

      
        self.mods = nn.ModuleList([layers.EncoderBlock(input_dim = hidden_size, num_heads = num_heads ,repeat = 2,k = 5) for i in range(7)])

        
        self.resizer = layers.FeedForward(hidden_size * 4,hidden_size,hidden_size)

        
        self.out = layers.Output(input_dim = hidden_size)

        

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs

        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        
        cw = self.word_emb(cw_idxs)

        qw = self.word_emb(qw_idxs)

        cc = self.char_emb(cc_idxs)

        qc = self.char_emb(qc_idxs)

        

        c_emb = self.emb(cw,cc)         # (batch_size, c_len, hidden_size)
        
        q_emb = self.emb(qw,qc)         # (batch_size, q_len, hidden_size)

       
     
        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, 2 * hidden_size)

        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, 2 * hidden_size)

        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)



        M0 = self.resizer(att)

        M0 = F.dropout(M0, p =self.dropout, training=self.training)

        
        for i, eblk in enumerate(self.mods):
             
            M0 = eblk(M0,c_mask)

        
        M1 = M0
     
        for i, eblk in enumerate(self.mods):
             
            M0 = eblk(M0,c_mask)

        
        M2 = M0

        M0 = F.dropout(M0, p=self.dropout, training=self.training)

        
        for i, eblk in enumerate(self.mods):
             
            M0 = eblk(M0,c_mask)

        M3 = M0

        p1,p2 = self.out(M1,M2,M3, c_mask)  # 2 tensors, each (batch_size, c_len)

       
        return p1,p2
