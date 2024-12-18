import torch
from pathlib import Path
from types import SimpleNamespace

NFT_Projects = ['Axies Infinity', 'Bored Ape Yacht Club', 'Crypto Kitties', 'Fat Ape Club', 'Roaring Leader'] 
nft_project_names = [''.join(Project_Name.split()).lower() for Project_Name in NFT_Projects]
min_purchase = [6, 2, 2, 0, 1]

Baseline_Methods = ['Greedy', 'Auction', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']
Breeding_Types = ['Heterogeneous', 'Homogeneous', 'ChildProject', 'None']
# Breeding_Types_Short = ['Heter', 'Homo', 'Child', 'None']


def default_args():
    args = SimpleNamespace()
    args.ckpt_dir = Path('ckpt')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device("cuda:0")
    args.breeding_topk = 10
    args.cand_lim = 50
    args.num_child_sample = 100
    args.mutation_rate = 0.1
    args.num_trait_div = 2
    args.num_attr_class = 2
    args.decay = 0.9
    args.eps = 1001

    args.setN = None
    args.setM = None
    args.no_method_cache = True
    args.overwrite = False
    args.gamma1 = 0.1
    args.gamma2 = 0.001
    return args

plot_colors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1770af', '#ADD8E6', '#BCBD22', '#D62728']