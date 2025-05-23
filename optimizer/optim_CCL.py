from optimizer.optim_Base import IROptimizer
from torch import nn
import torch
import math

class CCLOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]

        #   Modell Additional Parameter
        self.margin         = config["renyi_margin"]
        self.neg_weight     = config["neg_coefficient"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr}
        ])

    def cal_loss(self, y_pred):
        pos_logits = y_pred[:, 0]
        neg_logits = y_pred[:, 1:]

        pos_loss = torch.relu(1 - pos_logits)
        neg_loss = (torch.relu(neg_logits - self.margin)).mean(dim = 1) * self.neg_weight


        loss = (pos_loss + neg_loss).mean()

        return loss


    def regularize(self,users_emb, pos_emb, neg_emb):
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        return regularize

    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)


        loss            =  self.cal_loss(y_pred = y_pred)
        emb_loss        =  self.weight_decay * self.regularize(users_emb, pos_emb, neg_emb) / batch_size
        additional_loss =  self.model.additional_loss(
                                usr_idx = users.long(), 
                                pos_idx = pos.long(), 
                                embedding_user = embedding_user, 
                                embedding_item = embedding_item
                            )
        return loss, emb_loss + additional_loss

    def step(self, user, pos, neg):     
        CCL_loss,additional_loss = self.cal_loss_graph(user, pos, neg)
        loss = CCL_loss + additional_loss
        self.optimizer_descent.zero_grad()
        loss.backward()
        self.optimizer_descent.step()
        return CCL_loss.cpu().item()
    
    def save(self,path):
        all_states = self.model.state_dict()
        torch.save(obj = all_states, f = path)

