from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def tp_from_edge_id_lists(ref, pred) -> int:
    """Returns the number of shared elements between the
       two lists"""
    return len(set(ref).intersection(set(pred)))

def tp_at_thresholds(ref, pred_table):
    """
    Compute the true positives between ref and 
    pred_table['edge_id'] at the designated score thresholds
    given by pred_table.

    Returns a list of true positives at each prediction threshold
    pred_table: pd.DataFrame with |edge_id | score|
    """
    scores = sorted(set(pred_table['score']))
    tp_at_k = 0
    tp_at_k_plus_1 = 0
    # True positives
    tps = np.zeros(len(scores), dtype=int)
    # Positive predictions
    pps = np.zeros(len(scores), dtype=int)
    pps_at_k = 0
    pps_at_k_plus_1 = 0
    
    for k_plus_1, score in enumerate(scores):
        sel = pred_table['score'] == score
        subframe = pred_table.loc[sel]
        pred = subframe['edge_id'].values
        tp = tp_from_edge_id_lists(ref, pred)
        tp_at_k_plus_1 = tp + tp_at_k
        pps_at_k_plus_1 = len(pred) + pps_at_k
        tps[k_plus_1] = tp_at_k_plus_1
        pps[k_plus_1] = pps_at_k_plus_1
        tp_at_k = tp_at_k_plus_1
        pps_at_k = pps_at_k_plus_1
    df = pd.DataFrame({"threshold": scores, "tp": tps, "pp": pps})
    return df

example_ref = np.array(range(0, 3023)) * 10
example_pred = np.array(range(0, int(1e5))) * 100 
example_scores = example_pred 
example_pred_table = pd.DataFrame({"edge_id": example_pred,
    "score": example_scores})

