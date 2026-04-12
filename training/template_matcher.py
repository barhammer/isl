import numpy as np


class TemplateMatcher:
    def __init__(self, template_path, seq_len=30):
        self.template = np.load(template_path)  # (T, 127)
        self.seq_len = seq_len

        # take first valid window
        self.template_seq = self.template[:seq_len]

    def compare(self, live_seq):
        # shape: (30, 127)
        diff = np.abs(live_seq - self.template_seq)
        score = np.mean(diff)
        return score