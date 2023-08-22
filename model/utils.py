import torch
import torch.nn as nn
import numpy as np


def bds_to_seg_idx_ee(bds, seq_lens=None):
    '''
    Example:
        input[bds] = torch.LongTensor([[1,0,0,1,0,0],
                                        [1,0,0,0,0,0],
                                        [1,0,1,0,1,0]])
        seq_lens = torch.LongTensor([6,5,6])
        output[ee] = [tensor([3, 6]), tensor([5]), tensor([2, 4, 6])]
    '''
    idx = torch.arange(0, bds.size(1)).expand(bds.size(0),-1)
    idx = idx.to(bds.device)
    ee = bds*idx
    if seq_lens is not None:
        ee[:,-1]=seq_lens
    else:
        ee[:,-1]=bds.size(1)
    ee = [e[e>0] for e in ee]
    return ee

def return_des_lens_matrix(prd_bds, seq_lens):
    tgt_len = prd_bds.size(1)
    ee_s = bds_to_seg_idx_ee(prd_bds, seq_lens)
    ll_=[]
    for ee in ee_s:
        ll = torch.LongTensor([ee[0]]+[ee[i]-ee[i-1] for i in range(1,len(ee))])
        ll = np.hstack([np.arange(e,0,-1) for e in ll])
        ll = np.hstack([ll, [0]*(tgt_len-len(ll))])
        ll[ll<1] = 1
        ll_.append(list(ll))
    return torch.LongTensor(ll_).to(prd_bds.device)

def return_crs_attn_filter(enc_attn, src_bds, prd_bds, src_seq_lens=None, prd_seq_lens=None):
    # enc_attn (batch_size, prd_len, src_len)
    batch_size = src_bds.size(0)
    prd_len = prd_bds.size(1)
    src_len = src_bds.size(1)

    src_ee_s = bds_to_seg_idx_ee(src_bds, src_seq_lens)
    prd_ee_s = bds_to_seg_idx_ee(prd_bds, prd_seq_lens)

    fff = []
    for be, (src_ee, prd_ee) in enumerate(zip(src_ee_s, prd_ee_s)): # batch_size
        ff = []
        src_e0, prd_e0 = 0, 0
        for src_e1, prd_e1 in zip(src_ee, prd_ee): # seg_num
            f = torch.mean(enc_attn[be, :, src_e0:src_e1], dim=-1)
            # prescale
            (m,_), (M,_) = torch.min(f, dim=-1), torch.max(f, dim=-1)
            f = (f-m)/(M-m+1e-8)*(99) + 1
            ff.append(f.expand(prd_e1-prd_e0, -1))
            src_e0, prd_e0 = src_e1, prd_e1
        ff = torch.cat(ff) # (prd_len, src_len)
        fff.append(ff)
    fff = torch.stack(fff) # (batch_size, prd_len, src_len)
    return fff
  
def return_crs_attn_filter_eval(enc_attn_, src_bd_e1, src_bd_e2):
  # zero-batch supposed.
  # enc_attn_ (prd_len, src_len)

  f = torch.mean(enc_attn_[:, src_bd_e1:src_bd_e2], dim=-1)
  # prescale
  (m,_), (M,_) = torch.min(f, dim=-1), torch.max(f, dim=-1)
  f = (f-m)/(M-m+1e-8)*(99) + 1

  return f.unsqueeze(0)