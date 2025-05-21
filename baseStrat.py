from typing import List, Dict, Optional


class baseStrat:
    def __init__(self, name:str, model, base_dataset):
        self.name = name
        self.strategy = None
    

    def select(self, indices_to_score:List[int]):


