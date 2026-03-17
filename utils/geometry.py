import numpy as np

def get_centroid(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))