import numpy as np

from .utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """



    motifs_idx = []
    motifs_dist = []

    mp = np.copy(matrix_profile['mp'])
    mpi = np.copy(matrix_profile['mpi'])
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):

        motif_idx = np.argmin(mp)
        motif_dist = mp[motif_idx]


        if np.isnan(motif_dist) or np.isinf(motif_dist):
            break

        idx_left = motif_idx
        idx_right = int(mpi[motif_idx])

        motifs_idx.append((idx_left, idx_right))
        motifs_dist.append(motif_dist)

        mp = apply_exclusion_zone(mp, idx_left, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, idx_right, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
