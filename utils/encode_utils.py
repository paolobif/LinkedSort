import numpy as np
import pandas as pd


def find_distaces(manual, automated):
    """Returns list of distances"""
    distances = np.sqrt(np.square(automated[:, 0] - manual[0]) + np.square(automated[:, 1] - manual[1]))
    return distances


def find_best_matches(sort_xy, tracked_xy, min_distance=40):
    """Takes list of center xys form sort and center xys from tracked then
    compares the distance between each point. If the distance is less than 40
    pixes then it is a match. Returns df of tracked_xy with matches and non matches.

    Args:
        sort_xy (list[][x, y]): list of center xys from sort
        tracked_xy (list[][x, y]): list of center xys from raw outputs.
        min_distance (int, optional): min distance for match. Defaults to 30.

    Returns:
        pandas Df: df of tracked_xy with matches and non matches.
    """
    rows = []  # Stores dataframe rows.
    not_used = []  # Stores not used sort_xy idxs.

    for i, xy in enumerate(tracked_xy):
        distances = find_distaces(xy, sort_xy)
        min_idx = np.argmin(distances)
        a_loc = sort_xy[min_idx]

        if distances[min_idx] < min_distance:
            match = True
        else:
            match = False
            not_used.append(i)

        row = {"sort": a_loc, "tod": xy, "idx": i, "match": match, "distance": distances[min_idx]}
        rows.append(row)

    return pd.DataFrame(rows), not_used


