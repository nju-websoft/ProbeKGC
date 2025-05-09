from typing import Dict, List, Tuple
from knowledge_graph import KnowledgeGraph

_HEAD_PREDICTION = "head prediction"
_TAIL_PREDICTION = "tail prediction"



class Prompt:
    def __init__(self, data: Dict, kg: KnowledgeGraph) -> None:
        self.data = data
        self.kg = kg

        self.head, self.relation, self.tail = self.data["triple"]
        self.task = self.data["task"]
        assert self.task in [_HEAD_PREDICTION, _TAIL_PREDICTION], f"unknown task: {self.task}"
        
        if self.task == _HEAD_PREDICTION:
            self.query_entity = self.tail
            self.target_entity = self.head
        else:
            self.query_entity = self.head
            self.target_entity = self.tail
    
    def _triples2text(self, triples):
        statements = []
        for h, r, t in triples:
            template = self.kg.rel2text[r]["statement"]
            h_name = self.kg.ent2name[h]
            t_name = self.kg.ent2name[t]
            stat = template.replace("<head entity>", h_name).replace("<tail entity>", t_name)
            statements.append(stat)
        return "\n".join(statements)

    def _options2text(self, fixed_id=None):
        option_ids = ["A", "B", "C", "D"]
        option_id2ent = {option_id: self.data[option_id] for option_id in option_ids}
        if fixed_id is not None:
            gold_id = None
            for option_id in option_ids:
                if option_id2ent[option_id] == self.target_entity:
                    gold_id = option_id
                    break
            if gold_id is None:
                print(self.target_entity)
                print(option_id2ent)
                assert 0
            
            if gold_id != fixed_id:
                option_id2ent[gold_id] = option_id2ent[fixed_id]
                option_id2ent[fixed_id] = self.target_entity
            
            assert option_id2ent[fixed_id] == self.target_entity
                
        return "\n".join([f"{id}. {self.kg.ent2name[ent]}" for id, ent in option_id2ent.items()])

    def question_prompt(
        self,
        return_one: bool = True,
        use_cot: bool = True, 
        use_context: bool = False,
        fixed_option: str = None,
    ):
        ent_name = self.kg.ent2name[self.query_entity]
        ent_desc = self.kg.ent2desc[self.query_entity]
        rel_name = self.kg.rel2name[self.relation]

        if self.task == _HEAD_PREDICTION:
            triple = f"(<mask>, {rel_name}, {ent_name})"
            question = self.kg.rel2text[self.relation]["head prediction"].replace("<tail entity>", ent_name)
            entity_triples = self.kg.get_entity_triples(self.query_entity, self.relation, 1, 10)
        else:
            triple = f"({ent_name}, {rel_name}, <mask>)"
            question = self.kg.rel2text[self.relation]["tail prediction"].replace("<head entity>", ent_name)
            entity_triples = self.kg.get_entity_triples(self.query_entity, self.relation, 0, 10)
        entity_triple_text = self._triples2text(entity_triples)

        relation_triples = self.kg.get_relation_triples(self.relation, num=10)
        relation_triple_text = self._triples2text(relation_triples)

        options = self._options2text(fixed_option)

        system_prompt = f"""You are a good assistant about knowledge graph completion, which aims to predict the missing entity "<mask>" in an incomplete triple like (head entity, relation, <mask>) or (<mask>, relation, tail entity). Answer the question based on the given context."""
        prompt = f"""The incomplete triple is:\n{triple}\n\n"""
        
        if use_context:
            if len(ent_desc) > 0:
                prompt += f"""The background knowledge of entity "{ent_name}" is:\n{ent_desc}\n\n"""
            prompt += f"""The triples related to entity "{ent_name}" are:\n{entity_triple_text}\n\n"""
            prompt += f"""The triples related to relation "{rel_name}" are:\n{relation_triple_text}\n\n"""
        
        prompt += f"""Complete the triple by answering the following question:\n{question}\n{options}\n\n"""

        if use_cot:
            prompt += f"""Reasoning step by step based on the provided context. Finish the reasoning with the format "**The answer is <ID>.**", where <ID> is the option ID A/B/C/D.""" 
        else:
            prompt += f"Reply with the correct option ID (A/B/C/D). Do not reply any other words."
        
        if return_one:
            return system_prompt + "\n\n" + prompt
        else:
            return system_prompt, prompt
