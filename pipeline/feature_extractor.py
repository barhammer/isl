class FeatureExtractor:
    def extract(self, hand_landmarks_list):
        """
        hand_landmarks_list: list of MediaPipe hand landmarks for ONE person
        returns: fixed-length feature vector (2 hands max)
        """
        features = []

        hands = hand_landmarks_list if hand_landmarks_list else []

        for i in range(2):
            if i < len(hands):
                for lm in hands[i].landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)

        return features