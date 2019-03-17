import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from util import InputEmbeddings
import layers
import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

class Fusion(nn.Module):
    """ Implement heuristic matching trick from the paper
    """
    def __init__(self, input_dim, hidden_dim):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(input_dim * 4, hidden_dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        return self.tanh(self.linear(z))

class FusionLayer(nn.Module):
    """
    Heuristic matching trick from paper + highway.

    """

    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.linear_f = nn.Linear(input_dim * 4, input_dim, bias=True)
        self.linear_g = nn.Linear(input_dim * 4, 1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        gated = self.sigmoid(self.linear_g(z))
        fusion = self.tanh(self.linear_f(z))
        return gated * fusion + (1 - gated) * x


class BilinearSeqAtt(nn.Module):
    """ Combines the paragraph and question processing pipelines at the very end.

    This is used to implement equations 20 and 21.
    """
    def __init__(self, input_dim1, input_dim2):
        super(BilinearSeqAtt, self).__init__()
        self.linear = nn.Linear(input_dim1, input_dim2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x Tensor: shape (input_dim1, input_dim1)
        y Tensor: shape (input_dim2, input_dim2)
        """
        # breakpoint()
        xW = self.linear(x)
        # b * len
        xWy = torch.bmm(y, xW.unsqueeze(2)).squeeze(2)
        return xWy


class LinearAlign(nn.Module):
    """ Linear Align on the question side, eq (18) and (19)
    """
    def __init__(self, input_dim):
        super(LinearAlign, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        # breakpoint()
        Wx = self.linear(x)
        # eq (18)
        gamma = F.softmax(Wx, dim=2)

        # eq (19)
        q = torch.sum(Wx * gamma, dim=1)
        return q

class FusedSelfAttention(nn.Module):
    """ Self Attention on the paragraph side, eq (14) and (15).

    This is a bilinear self-alignment attention function. The implementation below is based on AllenNLP's
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/matrix_attention/bilinear_matrix_attention.py
    """

    def __init__(self, input_dim):
        super(FusedSelfAttention, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.weights)
        self.fuse = Fusion(input_dim, input_dim)

    def forward(self, x: torch.Tensor):
        intermediate = torch.matmul(x, self.weights)
        final = torch.matmul(intermediate, x.transpose(1,2))
        activation = F.softmax(x, dim=2)
        activated = activation * x
        fused = self.fuse(x, activated)

        return fused

class AlignedAttention(nn.Module):
    """Aligned attention for SLQA, adapted from BiDAF one.

    Computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is (c2q_attention, q2c_attention). This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape two tensors:
    - context (batch_size, context_len, 2 * hidden_size)
    - question (batch_size, question_len, 2 * hidden_size)


    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(AlignedAttention, self).__init__()
        self.drop_prob = drop_prob
        self.p_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.p_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p, q, p_mask, q_mask):
        batch_size, p_len, _ = p.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(p, q)        # (batch_size, p_len, q_len)
        p_mask = p_mask.view(batch_size, p_len, 1)  # (batch_size, p_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        
        # approx eq(4)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, p_len, q_len)

        # approx eq(5)
        # (bs, p_len, q_len) x (bs, q_len, hid_size) => (bs, p_len, hid_size)
        p_tilde = torch.bmm(s1, q)

        # approx eq(6)
        s2 = masked_softmax(s, p_mask, dim=1)       # (batch_size, p_len, q_len)
        # approx eq(7)
        q_tilde = torch.bmm(s2.transpose(1,2), p) # (bs, q_len, hid_size)

        x = (p_tilde, q_tilde)  

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        p_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, p_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, p_len, q_len)
        s0 = torch.matmul(c, self.p_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, p_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class SLQA(nn.Module):
    """ Implement "Multi-Granularity Hierarchical Attention Fusion
    Networksfor Reading Comprehension and Question Answering" from
    http://www.aclweb.org/anthology/P18-1158

    """
    def __init__(self, embeddings:InputEmbeddings, hidden_size, drop_prob=0.):
        super(SLQA, self).__init__()
        word_vectors = embeddings.word_vectors
        char_vectors = embeddings.char_vectors

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = AlignedAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.p_fusion1 = FusionLayer(2*hidden_size)

        self.q_fusion1 = FusionLayer(2*hidden_size)

        self.p_enc_eq_13 = layers.RNNEncoder(input_size= 2 * hidden_size,
                                             hidden_size = hidden_size,
                                             num_layers= 1,
                                             drop_prob=drop_prob)

        self.q_enc_eq_13 = layers.RNNEncoder(input_size= 2 * hidden_size,
                                             hidden_size = hidden_size,
                                             num_layers= 1,
                                             drop_prob=drop_prob)
        
        self.self_attention = FusedSelfAttention(2 * hidden_size)

        self.p_enc_eq_17 = layers.RNNEncoder(input_size= 2 * hidden_size,
                                             hidden_size = hidden_size,
                                             num_layers= 1,
                                             drop_prob=drop_prob)
        self.q_enc_eq_17 = layers.RNNEncoder(input_size= 2 * hidden_size,
                                             hidden_size = hidden_size,
                                             num_layers= 1,
                                             drop_prob=drop_prob)

        self.q_linear_align_18 = LinearAlign(2 * hidden_size)

        self.bilinear_start = BilinearSeqAtt(2*hidden_size, 2*hidden_size)
        self.bilinear_end = BilinearSeqAtt(2*hidden_size, 2*hidden_size)

    def forward(self, pw_idxs: torch.Tensor, pc_idxs: Optional[torch.Tensor], qw_idxs: torch.Tensor, qc_idxs: Optional[torch.Tensor]):
        """ Run a forward step
            pw_idxs: word indices in the paragraph  64, 254
            pc_idxs: char indices in the paragraph  64, 254, 16
            qw_idxs: word indices in the question 64, 20
            qc_idx: char indices in the question  64, 20, 16
        """

        p_mask = torch.zeros_like(pw_idxs) != pw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        p_len, q_len = p_mask.sum(-1), q_mask.sum(-1)

        p_emb = self.emb(pw_idxs, pc_idxs) # (batch_size, p_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs) # (batch_size, q_len, hidden_size)

        # eq (1)
        p_enc = self.enc(p_emb, p_len)    # (batch_size, p_len, 2 * hidden_size)

        # approx eq (2)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # approx eq (3)-(7)
        (p_tilde, q_tilde) = self.att(p_enc, q_enc,
                       p_mask, q_mask)    # 2 x tensors (batch_size, p_len, 2*hidden_size) 

        # eq (8) + (11)
        p_fused1 = self.p_fusion1(p_enc, p_tilde)

        # eq (9) + (12)
        q_fused1 = self.q_fusion1(q_enc, q_tilde)

        # eq (13)
        p_enc_13 = self.p_enc_eq_13(p_fused1, p_len)
        q_enc_13 = self.q_enc_eq_13(q_fused1, q_len)

        p_fused_16 = self.self_attention(p_enc_13)
        # more steps missing in here
        contextual_p = self.p_enc_eq_17(p_fused_16, p_len)

        # question partial processing        
        # eq (19)
        q_enc_17 = self.q_enc_eq_17(q_enc_13, q_len)
        weighted_q = self.q_linear_align_18(q_enc_17)

        logits_start = self.bilinear_start(weighted_q, contextual_p)
        logits_end = self.bilinear_end(weighted_q, contextual_p)        
        log_start = masked_softmax(logits_start, p_mask, log_softmax=True)
        log_end = masked_softmax(logits_end, p_mask, log_softmax=True)

        out = (log_start, log_end)
        return out
