import math
import numpy as np


# Finger chains (MediaPipe indices)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],      # Thumb
    [0, 5, 6, 7, 8],      # Index
    [0, 9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20]   # Pinky
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

    # Convert to numpy array (21 x 3)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    bone_vectors = []
    angles = []

    # ----------------------------------------
    # 🔹 Bone vectors + angles per finger
    # ----------------------------------------
    for chain in FINGER_CHAINS:
        # Bone vectors
        for i in range(len(chain) - 1):
            v = pts[chain[i + 1]] - pts[chain[i]]
            v = unit_vector(v)
            bone_vectors.extend(v)

        # Angles between consecutive bones
        for i in range(len(chain) - 2):
            v1 = unit_vector(pts[chain[i + 1]] - pts[chain[i]])
            v2 = unit_vector(pts[chain[i + 2]] - pts[chain[i + 1]])
            angles.append(angle_between(v1, v2) / math.pi)

    # ----------------------------------------
    # 🔹 Global orientation features (NEW)
    # ----------------------------------------

    # Palm direction: wrist → middle base (0 → 9)
    palm_vector = unit_vector(pts[9] - pts[0])

    # Hand span: index base → pinky base (5 → 17)
    span_vector = unit_vector(pts[17] - pts[5])

    global_features = list(palm_vector) + list(span_vector)

    return bone_vectors + angles + global_features


def landmarks_to_vector(hands):
    left_features = None
    right_features = None

    # ----------------------------------------
    # 🔹 Separate left and right hands
    # ----------------------------------------
    for hand in hands:
        features = extract_features(hand)

        label = getattr(hand, "label", None)

        if label == "Left":
            left_features = features
        elif label == "Right":
            right_features = features
        else:
            # fallback if label missing
            if left_features is None:
                left_features = features
            else:
                right_features = features

    # ----------------------------------------
    # 🔹 Determine feature size
    # ----------------------------------------
    if left_features is not None:
        feature_len = len(left_features)
    elif right_features is not None:
        feature_len = len(right_features)
    else:
        feature_len = 81  # fallback

    # ----------------------------------------
    # 🔹 Pad missing hands
    # ----------------------------------------
    if left_features is None:
        left_features = [0.0] * feature_len

    if right_features is None:
        right_features = [0.0] * feature_len

    # ----------------------------------------
    # 🔹 Final vector
    # ----------------------------------------
    vector = left_features + right_features

    return vector