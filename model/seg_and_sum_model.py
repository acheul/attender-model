import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import numpy as np
import math
from .bart_modified.modeling_bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast, BartConfig
from .bart_config import bart_config
from .positional_encoding import PositionalEncoding
from .utils import return_crs_attn_filter_eval

from easydict import EasyDict as edict

args = edict({'DEVICE': device,
              'MAX_ORG_LEN': 920,
              'MAX_ORG_SEG_LEN': 380,
              'MAX_SUM_LEN':160,
              'BATCH_SIZE': 8,
              'BOS': 0,
              'EOS': 1,
              })

kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')


class SegandSum(nn.Module):
    def __init__(self, dropout=0.1, BOS=args.BOS, EOS=args.EOS):
        super(SegandSum, self).__init__()
        bart_pre = BartForConditionalGeneration(config=bart_config)
        self.BOS = BOS
        self.EOS = EOS

        self.encoder = bart_pre.get_encoder()
        self.decoder = bart_pre.get_decoder()
        self.lm_head = bart_pre.lm_head

        self.positional_encoding = PositionalEncoding(d_model=768, dropout=dropout, max_len=1800)
        self.bd_enc_layerbd_enc_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=768, batch_first=True, dropout=dropout)
        self.bd_enc = nn.TransformerEncoder(self.bd_enc_layerbd_enc_layer, num_layers=2)
        self.bd_prj_head = nn.Linear(768, 2)

    def batch_bd_ee(self, enc_last, src_masks=None, max_seg_num=10, min_seg_num=2, min_seg_len=20, min_prd_value=0.2):
        '''
        return boundary output for each source input.
          * this method works implicit here, but can be modified into explicit way.
        Input argments:
          enc_last: last output of encoder
          max_seg_num: maximum number of total segments
          min_prd_value: minimum value point determined to be a boundary. (all predicted values are between 0 and 1)
        '''
        bd_prd = self.positional_encoding(enc_last)
        bd_prd_mask = (1-src_masks).to(torch.bool) if src_masks is not None else None
        bd_prd = self.bd_enc(bd_prd, src_key_padding_mask=bd_prd_mask)
        bd_prd = self.bd_prj_head(bd_prd)
        bd_prd = nn.functional.softmax(bd_prd, dim=-1)
        bd_prd = bd_prd[:,:,1]

        sort_, argsort_ = torch.sort(bd_prd, dim=-1, descending=True)
        argsort_[:,:max_seg_num]

        batch_bd_ee = []
        for sort_value, argsort_idx in zip(sort_, argsort_):
            bd_ee = [0,]
            for s,a in zip(sort_value, argsort_idx):
                if len(bd_ee)>max_seg_num:
                    break
                if a!=0:
                    if len(bd_ee)<min_seg_num or s>=min_prd_value:
                        if len([True for e in bd_ee if torch.abs(a-e)<min_seg_len])==0: 
                            bd_ee.append(int(a))
                    else:
                        break
            batch_bd_ee.append(sorted(bd_ee))
        return batch_bd_ee, bd_prd

    def seg_and_sum(self, src_ids, src_masks,
                        des_lens=[None], max_sum_length=500,
                        max_seg_num=10, min_seg_num=2, min_seg_len=20, min_prd_value=0.2
                        ):
        '''
        return segmental summaries.
          * this method is supposed to be one-size batch processing *
        inputs argments:
          src_ids: input-token-ids # (1=batch_size, input_length)
          src_masks: input-token-masks
          des_lens: a list of desired token-length for each segments # [4, 5, 6, ...]; [None, ...]
          max_seg_num: maximum number of total segments
        '''
        batch_size = src_ids.size(0)

        # encoder
        oo = self.encoder(src_ids, src_masks, output_attentions=True)
        enc_last = oo.last_hidden_state
        enc_attn = oo.attentions
        enc_attn = torch.stack(enc_attn).mean(dim=(0,2)) # (batch_size, src_size, src_size)

        # get boundaries
        src_bd_ee, bd_prd = self.batch_bd_ee(enc_last, src_masks, max_seg_num, min_seg_num, min_seg_len, min_prd_value)
        src_bd_ee = src_bd_ee[0]+[src_ids.size(1)+1]
        src_bd_e1e2_list = [(e1,e2) for e1,e2 in zip(src_bd_ee[:-1], src_bd_ee[1:])]

        if len(des_lens)<len(src_bd_e1e2_list):
            des_lens = des_lens * len(src_bd_e1e2_list)

        # decoding
        builds = torch.LongTensor([args.BOS]).expand(batch_size, -1).to(device)
        current = builds[:,-1:]
        past = None

        seg_n = 0
        src_bd_e1e2 = src_bd_e1e2_list[seg_n]
        des_len_ = des_lens[seg_n]
        crs_attn_filter = return_crs_attn_filter_eval(enc_attn[0], src_bd_e1e2[0],src_bd_e1e2[1])
        crs_attn_filter = crs_attn_filter.unsqueeze(0)

        score_s = [0]*len(src_bd_e1e2_list)

        t=0
        while t<max_sum_length:
            t+=1

            if des_len_ is not None:
                des_lens_matrix = torch.LongTensor([des_len_]).view(1,-1).contiguous().to(device)
            else:
                des_lens_matrix = None

            dd = self.decoder(input_ids=current, attention_mask=None,
                              encoder_hidden_states=enc_last, encoder_attention_mask=src_masks,
                              cross_attention_filter=crs_attn_filter,
                              des_lengths=des_lens_matrix,
                              past_key_values=past,
                              )

            if des_len_ is not None:
                if des_len_ > 1:
                    des_len_ -= 1

            dec_last = dd.last_hidden_state
            logits = self.lm_head(dec_last)
            probs = nn.functional.softmax(logits, dim=-1)
            past = dd.past_key_values

            score, current = torch.max(probs[:,-1:], dim=-1)
            builds = torch.cat((builds, current), dim=-1)
            score_s[seg_n] += score[0].detach()

            if current[0][0]==self.BOS:
                seg_n += 1
                if seg_n<len(src_bd_e1e2_list):
                    src_bd_e1e2 = src_bd_e1e2_list[seg_n]
                    des_len_ = des_lens[seg_n]
                    crs_attn_filter = return_crs_attn_filter_eval(enc_attn[0], src_bd_e1e2[0],src_bd_e1e2[1])
                    crs_attn_filter = crs_attn_filter.unsqueeze(0)
                else:
                    break

        score_s = [s/l if l is not None else s/10 for s,l in zip(score_s, des_lens)]

        return builds, src_bd_ee, bd_prd, score_s




