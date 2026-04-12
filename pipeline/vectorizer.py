import math
import numpy as np


# MediaPipe hand connections (important pairs)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]


def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)
    return math.acos(dot)


def extract_features(hand):
    landmarks = hand.landmark

    # convert to numpy
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # ----------------------------------------
    # 🔹 Bone vectors
    # ----------------------------------------
    bone_vectors = []

    for (i, j) in CONNECTIONS:
        v = pts[j] - pts[i]
        v = unit_vector(v)
        bone_vectors.extend(v)

    # ----------------------------------------
    # 🔹 Angles between consecutive bones
    # ----------------------------------------
    angles = []

    for k in range(len(CONNECTIONS) - 1):
        i1, j1 = CONNECTIONS[k]
        i2, j2 = CONNECTIONS[k + 1]

        v1 = unit_vector(pts[j1] - pts[i1])
        v2 = unit_vector(pts[j2] - pts[i2])

        ang = angle_between(v1, v2)
        angles.append(ang)

    return bone_vectors + angles


def landmarks_to_vector(hands, confidence):
    vector = []

    for hand in hands[:2]:
        features = extract_features(hand)
        vector.extend(features)

    # pad if less than 2 hands
    target_len = len(extract_features(hands[0])) if hands else 75

    while len(vector) < target_len * 2:
        vector.extend([0.0] * target_len)

    vector.append(confidence)

    return vector