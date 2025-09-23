import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    if len(ts1) != len(ts2):
        raise ValueError("Time series must have the same length")

    squared_diff = (ts1 - ts2) ** 2
    ed_dist = np.sqrt(np.sum(squared_diff))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    if len(ts1) != len(ts2):
        raise ValueError("Time series must have the same length")

    n = len(ts1)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    dtw_dist = D[n, n]

    return dtw_dist
