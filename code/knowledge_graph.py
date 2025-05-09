import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List
from collections import Counter, defaultdict
from functools import partial, cmp_to_key


def load_triples(file_path: str):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples


class Relation_Co:
    def __init__(self, triples: List) -> None:
        self.triples = triples
        self.rel = self.get_relations()
        self.one_hop_triples, self.one_hop_relations = self.get_one_hop_triples()
        self.rel_co = self.count_rel_co()

    def get_relations(self):
        rel = set()
        for h, r, t in self.triples:
            rel.add(r)
        return rel
    
    def get_one_hop_triples(self):
        one_hop_triples = defaultdict(set)
        one_hop_relations = defaultdict(set)
        for h, r, t in self.triples:
            one_hop_triples[h].add((h, r, t))
            one_hop_triples[t].add((h, r, t))
            one_hop_relations[h].add((r, 0))
            one_hop_relations[t].add((r, 1))
        return one_hop_triples, one_hop_relations

    def count_rel_co(self):
        rel_co = defaultdict(int)
        for entity, one_hop_triples in self.one_hop_triples.items():
            for h, r, t in one_hop_triples:
                for r_sample, direct in self.one_hop_relations[entity]:
                    if r == r_sample:
                        continue
                    elif h == entity:
                        rel_co[((r, 0) , (r_sample, direct))] += 1
                    else:
                        rel_co[((r, 1) , (r_sample, direct))] += 1
        return rel_co

    def get_rel_co(self, rel, direct):
        for r in self.rel:
            if r != rel:
                print(r, self.rel_co[(rel, direct), (r, 0)], self.rel_co[(rel, direct), (r, 1)])


class Relation_PMI:
    def __init__(self, triples: List, save_path: str):
        self.triples = triples
        
        self.rel_list = sorted(list(set([r for _, r, _ in triples])))
        self.rel_list += ["inv_"+r for r in self.rel_list]
        self.rel2idx = {rel: idx for idx, rel in enumerate(self.rel_list)}
        self.idx2rel = {idx: rel for rel, idx in self.rel2idx.items()}

        if os.path.exists(save_path):
            self.rel_pmi = np.load(save_path)
        else:
            self.rel_pmi = self._pmi_from_triples()
            np.save(save_path, self.rel_pmi)

    def _pmi_from_triples(self):
        ent_pairs = set([(h, t) for h, _, t in self.triples] + [(t, h) for h, _, t in self.triples])
        ent_pair2idx = {ent_pair: idx for idx, ent_pair in enumerate(ent_pairs)}
        print(len(ent_pair2idx))
        
        M = np.zeros((len(ent_pairs), len(self.rel_list)), dtype=int)
        for h, r, t in tqdm(self.triples):
            M[ent_pair2idx[(h, t)], self.rel2idx[r]] = 1
            M[ent_pair2idx[(t, h)], self.rel2idx["inv_"+r]] = 1
        M = torch.from_numpy(M).float().cuda()

        co_occurrence = torch.matmul(M.t(), M).cpu().numpy()
        count_r = torch.sum(M, dim=0).cpu().numpy()
        N = len(ent_pairs)
        PMI = np.zeros_like(co_occurrence, dtype=float)
        for i in tqdm(range(len(self.rel_list))):
            for j in range(len(self.rel_list)):
                if co_occurrence[i, j] == 0:
                    PMI[i, j] = 0
                else:
                    PMI[i, j] = max(np.log((co_occurrence[i, j] * N) / (count_r[i] * count_r[j])), 0)
        return PMI


class KnowledgeGraph:
    def __init__(
        self, 
        data_dir: str,
    ) -> None:
        ent2text = json.load(open(os.path.join(data_dir, 'entity.json'), 'r', encoding='utf-8'))
        self.rel2text = json.load(open(os.path.join(data_dir, 'relation.json'), 'r', encoding='utf-8'))
        self.ent2name = {ent: ent2text[ent]["name"] for ent in ent2text}
        self.ent2desc = {ent: ent2text[ent]["desc"] for ent in ent2text}
        if "ProbeFB" in data_dir:
            self.rel2name = {rel: rel.split(".")[-1] for rel in self.rel2text}
        elif "ProbeWN" in data_dir:
            self.rel2name = {rel: rel.replace("_", " ").strip() for rel in self.rel2text}
        elif "ProbeYG" in data_dir:
            self.rel2name = {rel: self.rel2text[rel]["name"] for rel in self.rel2text}
        else:
            assert 0, data_dir
        
        self.train_triplets = load_triples(os.path.join(data_dir, 'train.txt'))
        self.valid_triplets = load_triples(os.path.join(data_dir, 'valid.txt'))
        self.test_triplets = load_triples(os.path.join(data_dir, 'test.txt'))
        triplets = self.train_triplets + self.valid_triplets + self.test_triplets
        self.ent_list = sorted(list(set([h for h, _, _ in triplets] + [t for _, _, t in triplets])))
        self.rel_list = sorted(list(set([r for _, r, _ in triplets])))
        
        self.relation_co = Relation_Co(self.train_triplets)
        self.relatoin_pmi = Relation_PMI(self.train_triplets, os.path.join(data_dir, "relation_pmi.npy"))

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.relation2triples = defaultdict(list)
        for h, r, t in self.train_triplets:
            self.out_edges[h].append((h, r, t))
            self.in_edges[t].append((h, r, t))
            self.relation2triples[r].append((h, r, t))

        self.relation_triples = self._load_relation_triples(os.path.join(data_dir, "relation_triples.json"))

    def get_entity_triples(self, entity, relation, direct, num, sample_type: str="co"):
        if sample_type == "random":
            return self._entity_triples_random(entity, num)
        elif sample_type == "co":
            return self._entity_triples_co(entity, relation, direct, num)
        elif sample_type == "pmi":
            pass
        else:
            assert 0, sample_type

    def _entity_triples_co(self, ent, rel, direct, num):
        out_edges = self.out_edges[ent]
        out_scores = [self.relation_co.rel_co[(r, 0), (rel, direct)] for h, r, t in out_edges]
        out_sorted_indices_desc = np.argsort(out_scores)[::-1]

        in_edges = self.in_edges[ent]
        in_scores = [self.relation_co.rel_co[(r, 1), (rel, direct)] for h, r, t in in_edges]
        in_sorted_indices_desc = np.argsort(in_scores)[::-1]

        if num <= len(out_edges):
            return [out_edges[out_sorted_indices_desc[i]] for i in range(num)]
        elif num <= len(out_edges + in_edges):
            return out_edges + [in_edges[in_sorted_indices_desc[i]] for i in range(num - len(out_edges))]
        else:
            edges = out_edges + in_edges
            return edges

    def _entity_triples_random(self, ent, num):
        out_edges = self.out_edges[ent]
        in_edges = self.in_edges[ent]

        if num <= len(out_edges):
            return random.sample(out_edges, num)
        elif num <= len(out_edges + in_edges):
            return random.sample(out_edges + in_edges, num)
        else:
            edges = out_edges + in_edges
            random.shuffle(edges)
            return edges

    def get_relation_triples(self, relation, num=10, is_random=False):
        if is_random:
            return random.sample(self.relation2triples[relation], num)
        return self.relation_triples[relation][: num]

    def _load_relation_triples(self, save_path: str):
        if os.path.exists(save_path):
            return json.load(open(save_path, 'r', encoding='utf-8'))

        relation2triples = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for h, r, t in self.train_triplets:
            relation2triples[r].append((h, r, t))
            out_degree[h] += 1
            in_degree[t] += 1

        relation_sample_triples = defaultdict(list)
        for relation in self.rel_list:
            triples = relation2triples[relation]
            ent2count = defaultdict(int)
            
            def cmp(triple1, triple2):
                h1, r1, t1 = triple1
                h2, r2, t2 = triple2
                
                cnt1, cnt2 = ent2count[h1]+ent2count[t1], ent2count[h2]+ent2count[t2]
                if cnt1 < cnt2:
                    return -1
                elif cnt1 > cnt2:
                    return 1
                else:
                    deg1 = in_degree[h1] + out_degree[h1] + in_degree[t1] + out_degree[t1]
                    deg2 = in_degree[h2] + out_degree[h2] + in_degree[t2] + out_degree[t2]
                    if deg1 >= deg2:
                        return -1
                    else:
                        return 1

            demo_list = []
            while len(demo_list) < 50 and len(triples) > 0:
                triples = sorted(triples, key=cmp_to_key(cmp))
                h, r, t = triples.pop(0)

                ent2count[h] += 1
                ent2count[t] += 1
                demo_list.append((h, r, t))
            relation_sample_triples[relation] = demo_list
        
        json.dump(relation_sample_triples, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        return relation_sample_triples
