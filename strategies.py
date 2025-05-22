import torch
from typing import List
import numpy as np
from tqdm.auto import tqdm

"""
The model output a list of size 1 (because we do it image by image)


each element (the only one) of the list is a Dict with the following keys :

    - boxes 

"""


@torch.no_grad()
def random_selection(model, base_ds, unlabeled_indices: List[int], budget: int):
    return np.random.choice(unlabeled_indices, budget, replace=False)

def tanguy_entropy(p:float):
    assert p > 0
    assert p < 1

    N = 20
    return -p*np.log(p) - (1-p) * np.log((1-p)/(N-1)) 

@torch.no_grad()
def entropy_selection(model, base_ds, unlabeled_indices: List[int], budget: int):
    strat_model = model.cuda(0)
    strat_model.eval()

    entropies = np.zeros(len(unlabeled_indices))

    
    for i,index in tqdm(enumerate(unlabeled_indices),total=len(unlabeled_indices) ,desc = "Scoring with entropies"):
        image = base_ds[index]["image"].unsqueeze(0).cuda()
        output = strat_model(image)
        # output is a list of size 1
        output = output[0]
        # get the scores
        scores = output["scores"].cpu().numpy()
        # get the boxes
        boxes = output["boxes"].cpu().numpy()
        # get the labels
        labels = output["labels"].cpu().numpy()

        # compute the entropy
        if len(boxes) == 0:
            entropies[i] = 0
        else:
            detection_entropies = np.array([tanguy_entropy(i) for i in scores])
            entropies[i] = detection_entropies.mean()

    print(f"{entropies.mean()=} {entropies.max()=} {entropies.min()=} {entropies.median()=}")
    return unlabeled_indices[np.argsort(entropies)[-budget:]] 


@torch.no_grad()
def new_selection(model, base_ds, unlabeled_indices: List[int], budget: int):
    pass


