import sys
import argparse
from dataset import prepare_nft_data

import time
import torch
import logging

from utils import set_seed, set_verbose, dumpj, deep_to_pylist
from solver import get_solver
from arguments import default_args, nft_project_names, Breeding_Types, Baseline_Methods


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', choices=[0,1,2], default=2) # warning; info; debug
    config = parser.parse_args()

    set_seed(config.seed)
    set_verbose(config.verbose)

    # prepare_nft_data()
    run_main_exp()

def run_main_exp():
    args = default_args()

    msg = f'''
    >>> Experimenting ...
        {nft_project_names}
        x {Breeding_Types}
        x {Baseline_Methods}\n'''
    print(msg)

    for nft_project_name in nft_project_names[::-1]:
        args.nft_project_name = nft_project_name
        for _method in Baseline_Methods:
            for _breed in Breeding_Types:
                result_file = args.ckpt_dir / f'{nft_project_name}_{_method}_{_breed}.json'
            
                if result_file.exists() and not args.overwrite:
                    logging.info(f'|> result file:{result_file} exists <|')
                else:
                    logging.info(f'...running [{nft_project_name}, {_method}, {_breed}] experiment...')
                    args.breeding_type = _breed
                    Solver = get_solver(args, _method)

                    start_time = time.time()
                    add_time = Solver.solve() 
                    add_time = 0 if add_time is None else add_time
                    Solver.count_results() 
                    runtime = time.time() - start_time + add_time
                    Result = {
                        'runtime': runtime,
                        'seller_revenue': Solver.seller_revenue,
                        'avg_buyer_utility': Solver.buyer_utilities.mean().item(),
                        'utility_component': Solver.utility_component
                    }
                    Result = {k:deep_to_pylist(v) for k, v in Result.items()}

                    dumpj(Result, result_file)
                    torch.save(
                        {'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing}, 
                        result_file.with_suffix('.pth')
                        )
                    print('______________________________________experiment done.')

if __name__ == '__main__':
    main()