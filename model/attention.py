import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention,self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)


  def masked_softmax(self, X, valid_lens):
    if valid_lens is None:
      return nn.functional.softmax(X, dim=-1)
    else:
      shape = X.shape
      if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
      else:
        valid_lens = valid_lens.reshape(-1)
      masked_value = -1e6
      X = X.reshape(-1, shape[-1])
      maxlen = X.size(1)
      mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:] < valid_lens[:,None]
      X[~mask] = masked_value
      X = X.reshape(shape)
      m, _ = torch.max(X, dim=-1)
      m, _ = torch.max(m, dim=-1)
      return nn.functional.softmax(X, dim=-1), m

  def forward(self, q, k, v, valid_lens=None, threshold=0.8):
    d = q.shape[-1]
    scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(d)
    self.attention_weights, self.max_score = self.masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), v), self.max_score * math.sqrt(d) > threshold

def masked_mean(X, valid_lens=None):
    # X : [bs, n, d]
    if valid_lens is None:
      return X.sum(dim=1)
    else:
      X = X.permute(0,2,1)
      shape = X.shape
      valid_lens = torch.repeat_interleave(valid_lens, shape[1])
      masked_value = 0
      X = X.reshape(-1, shape[-1])
      maxlen = X.size(1)
      mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:] < valid_lens[:,None]
      X[~mask] = masked_value
      valid_lens = valid_lens.reshape(shape[0], -1)
      return X.reshape(shape).sum(dim=-1)/valid_lens