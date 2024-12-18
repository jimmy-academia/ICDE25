import torch
from tqdm import tqdm
from .constant import make_batch_indexes
from .base import BaseSolver
from utils import *

class BANTERSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def solve(self):
        '''
        proposed BANTER method for NFT and pricing recommendation to obtain Market Equilibrium
        '''
        # self.pricing, self.holdings.
        ## pricing initialization
        # ablation_id 0: full, 1: no init, 2: only init
        # self.pricing_list = []
        self.args.ablation_id = 0
        if self.args.ablation_id in [0, 2]:
            self.pricing = self.greedy_init_pricing()
        else:
            self.pricing = torch.rand(self.nftP.M, device=self.args.device)

        ## demand-based optimization
        eps = self.args.eps
        pbar = tqdm(range(64), ncols=88, desc='BANTER Solver!') #64
        # self.pricing_list.append(self.pricing)
        if self.args.ablation_id == 2:
            self.holdings = self.solve_user_demand()
            self.count_results()
            return

        counter = 0
        best_counter = 0
        best_excess = 1e9
        best_pricing = None
        
        for iter_ in pbar:

            demand = self.solve_user_demand()
            demand = demand.sum(0)
            excess = demand - self.nft_counts

            eps = eps*torch.exp(-self.args.gamma1*excess.norm().sum()/self.nft_counts.sum() \
                + self.args.gamma2 * torch.tanh(self.ratio - 1))
            # elif self.args.schedule_id == 1:
                # eps = eps * 0.99

            self.pricing *= ( 1 +  eps * excess/(excess.abs().sum()))
            self.pricing = torch.where(self.pricing < 1e-10, 1e-10, self.pricing) 
            pbar.set_postfix(excess=float(excess.abs().sum()))

            if excess.abs().sum() < 500:
                counter += 1
            else:
                counter = 0
            if excess.abs().sum() < best_excess:
                best_excess = excess.sum()
                best_pricing = self.pricing.clone()
                best_counter = 0
            else:
                best_counter += 1
        
        self.pricing = best_pricing
        self.holdings = self.solve_user_demand()

    def solve_user_demand(self, set_user_index=None):

        div = 20 if self.nftP.N < 9000 else self.nftP.N // 500
        if set_user_index is not None:
            batch_user_iterator = [set_user_index]
        else:
            batch_user_iterator = self.buyer_budgets.argsort(descending=True).tolist()
            batch_user_iterator = make_batch_indexes(batch_user_iterator, self.nftP.N//div)

        spending = torch.rand(self.nftP.N, self.nftP.M+1).to(self.args.device) # N x M+1 additional column for remaining budget
        
        spending /= spending.sum(1).unsqueeze(1)

        pbar = tqdm(range(16), ncols=88, desc='Solving user demand!', leave=False)
        user_eps = 1e-4
        for __ in pbar:
            buyer_utility = 0
            for user_index in tqdm(batch_user_iterator, ncols=88, leave=False, total=div+1):
                spending_var = spending[user_index]
                spending_var.requires_grad = True
                
                batch_budget = self.buyer_budgets[user_index]
                holdings = self.hatrelu(spending_var[:, :-1]*batch_budget.unsqueeze(1)/self.pricing.unsqueeze(0)) #, self.nft_counts
                _utility = self.calculate_buyer_utilities(user_index, holdings, batch_budget, self.pricing)
                _utility.backward(torch.ones_like(_utility))

                buyer_utility += _utility.detach().mean().item()
                _grad = spending_var.grad
                spending[user_index] += user_eps* _grad
                # spending = torch.where(spending < 0, 0, spending)
                spending.clamp_(min=0)
                spending /= spending.sum(1).unsqueeze(1)
            
            pbar.set_postfix(delta= float(spending[:, -1].sum() - spending[:, -1].sum()))

        demand = self.hatrelu(spending[:, :-1]*self.buyer_budgets.unsqueeze(1)/self.pricing.unsqueeze(0)) #, self.nft_counts
        return demand

    def hatrelu(self, x, threshold=1):
        ## upperbound caps x at threshold
        return threshold - torch.nn.functional.relu(threshold-x)
