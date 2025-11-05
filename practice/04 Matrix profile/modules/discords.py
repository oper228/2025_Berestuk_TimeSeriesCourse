import numpy as np

from .utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp_values = np.copy(matrix_profile['mp'])
    mp_indices = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        discord_idx = np.argmax(mp_values)
        discord_dist = mp_values[discord_idx]
        nn_idx = mp_indices[discord_idx]

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

        mp_values = apply_exclusion_zone(mp_values, discord_idx, excl_zone, val=-np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
