import torch
from torch.utils.data import BatchSampler, Sampler
import random
import numpy as np

class DomainBalancedBatchSampler(BatchSampler):
    def __init__(self, domain_indices, n_samples_per_domain):
        self.domain_indices = domain_indices
        self.n_samples_per_domain = n_samples_per_domain
        self.domains = list(domain_indices.keys())
        
        self.n_domains = len(self.domains)
        self.batch_size = self.n_domains * self.n_samples_per_domain

        self.batch_size = self.n_domains * self.n_samples_per_domain
        # 以最大 domain 为基准，决定 batch 数量
        self.num_batches = max(len(v) for v in self.domain_indices.values()) // self.n_samples_per_domain

    def __iter__(self):
        # 先 shuffle 一下每个 domain 的 indices
        shuffled_domains = {
            d: random.sample(idxs, len(idxs)) for d, idxs in self.domain_indices.items()
        }
        # 在采样过程中循环遍历每个 domain 的样本索引
        for i in range(self.num_batches):
            batch = []
            for d in self.domains:
                domain_idxs = shuffled_domains[d]
                start = i * self.n_samples_per_domain
                end = (i + 1) * self.n_samples_per_domain
                # batch.extend(shuffled_domains[d][start:end])
                if end <= len(domain_idxs):
                    # 正常切片
                    selected = domain_idxs[start:end]
                else:
                    # oversampling: 不够就补采样
                    remain = domain_idxs[start:]  # 剩下的
                    need = self.n_samples_per_domain - len(remain)
                    extra = random.choices(domain_idxs, k=need)  # 有放回采样避免还是不够
                    selected = remain + extra
                batch.extend(selected)
            yield batch

    def __len__(self):
        return self.num_batches