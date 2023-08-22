import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import numpy as np
import math, argparse

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from model.seg_and_sum_model import (kobart_tokenizer, SegandSum)

parser = argparse.ArgumentParser()
parser.add_argument("--BOS", default=0)
parser.add_argument("--EOS", default=1)
parser.add_argument("--MAX_ORG_LEN", default=920)
parser.add_argument("--max_sum_length", default=400)
parser.add_argument("--max_seg_num", default=7)
parser.add_argument("--min_seg_num", default=2)
parser.add_argument("--min_seg_len", default=50)
parser.add_argument("--min_prd_value", default=0.8)
parser.add_argument("--DEVICE", default="cuda", help="cuda or cpu")
parser.add_argument("--state_path", default="../model/states/c34_1", help="state file path")
args = parser.parse_args() 

def upload_state_dict(state_path):
  model = SegandSum()
  model.load_state_dict(torch.load(state_path))
  return model

def do_seg_and_sum(model, script, des_lens=[None], max_sum_length=400, max_seg_num=10, min_seg_num=2, min_seg_len=50, min_prd_value=0.2):
  inputs = kobart_tokenizer.encode(script, return_tensors='pt', truncation=True, max_length=args.MAX_ORG_LEN)
  inputs = torch.cat([torch.LongTensor([args.BOS]*inputs.size(0)).unsqueeze(-1), inputs, torch.LongTensor([args.EOS]*inputs.size(0)).unsqueeze(-1)], dim=1)   
  inputs = inputs.to(device)
  print(inputs.size())
  
  builds, src_bd_ee, bd_prd, score_s = model.seg_and_sum(inputs, None,
                                        des_lens=des_lens,
                                        max_sum_length=max_sum_length, max_seg_num=max_seg_num, min_seg_num=min_seg_num, min_seg_len=min_seg_len, min_prd_value=min_prd_value)

  return builds, src_bd_ee, bd_prd, score_s, inputs

if __name__ == '__main__':
  model = upload_state_dict(args.state_path)
  while True:
    
    # retrieves .txt file then conducts Segmentaion and Summarization.
    # des_lens: desired lengths
    # command line example: ```test_src/news_1.txt, 15, 20, 25, None```
    script_path_and_des_lens = input()
    if script_path_and_des_lens=='END': # 종료문: 'END'
      break
    
    script_path = script_path_and_des_lens.split(", ")[0]
    des_lens = script_path_and_des_lens.split(", ")[1:]
    des_lens = [None if e=='None' else int(e) for e in des_lens]

    with open(script_path, "r", encoding="UTF-8") as file:
      script = file.read()
    
    builds, src_bd_ee, bd_prd, score_s, inputs = do_seg_and_sum(model, script, des_lens=des_lens,
                                                                max_sum_length=args.max_sum_length, max_seg_num=args.max_seg_num, min_seg_num=args.min_seg_num, min_seg_len=args.min_seg_len, min_prd_value=args.min_prd_value)
    
    print("SUMMARIES:")
    for e in builds:
        print(kobart_tokenizer.decode(e))
    print("BOUNDARY TOKEN IDS:", src_bd_ee)
    bd_prd = bd_prd.detach().cpu().numpy()
    print("BOUNDARY POINT VALUES:", bd_prd[0][src_bd_ee[:-1]])
    print("Prob scores:", score_s)
    print("SEGMENTS-Original script")
    for e,(e1,e2) in enumerate(zip(src_bd_ee[:-1], src_bd_ee[1:])):
        print(f"SEG{e+1}", kobart_tokenizer.decode(inputs[0, e1:e2]).replace("\n", "\\n"))