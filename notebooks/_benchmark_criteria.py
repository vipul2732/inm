from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _tp_from_edge_id_lists(ascending_predicted_edge_id_listlike,
                  ascending_reference_edge_id_listlike,
                  running_total=0) -> float:
    """
    1. Check the orderings
    2. Check the types (integer)
    3. Increase the minimal reference value such that it is the
       largest possible value while still being lt the predicted
       value.
    4. Select the minimal predicted value and compare to reference values
       until it is found or a greater value is found.
    5. Update the running total
    6. Rinse and repeat until finished
    """
    #Obtain the bounds
    refr_ll = ascending_reference_edge_id_listlike
    pred_ll = ascending_predicted_edge_id_listlike
    # If the reference is empty return the running total
    refr_len = len(refr_ll)
    pred_len = len(pred_ll)
    if (refr_len == 0) or (pred_len == 0):
        return running_total
    min_refr = refr_ll[0]
    max_refr = refr_ll[-1]
    min_pred = pred_ll[0]
    max_pred = pred_ll[-1]
    pred_edge_index = 0
    #If the prediction is outside the reference return the running total
    if (min_pred > max_refr) or (max_pred < min_refr):
        return running_total 

    # 1. Shrink the  prediction array
    for refr_indx, refr_edge_id in enumerate(refr_ll):
        tmp = pred_ll 
        tmp_len = len(tmp)
        if tmp_len  == 1:
            if refr_edge_id == tmp[0]:
                return running_total + 1
            else:
                return running_total
        for pred_indx, pred_edge_id in enumerate(tmp):
            if refr_edge_id >= pred_edge_id:
                if refr_edge_id == pred_edge_id:
                    running_total += 1
                if pred_indx == tmp_len - 1:
                    pred_ll = []
                else:
                    pred_ll = pred_ll[pred_indx + 1:]
                break
    return running_total 

ref = np.array(range(0, 3023)) * 10
pred = np.array(range(0, int(1e6))) * 4

def df2unordered_pair_set(df, colA='A', colB='B'):
    out = [] 
    for label, (a, b) in df.loc[:, [colA, colB]].iterrows(): 
        pair = UndirectedNonselfEdge(a, b)
        if pair not in out:
            out.append(pair)
    return set(out)   

_example = pd.DataFrame({'A': [0, 0, 1, 1, 2, 2,],
                         'B': [1, 2, 0, 2, 0, 1,]})
