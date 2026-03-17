import copy


class HandInterpolator:

    @staticmethod
    def get_center(hand):
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    @staticmethod
    def get_velocity(prev, curr):
        total = 0.0
        count = 0

        for i in range(len(curr.landmark)):
            dx = curr.landmark[i].x - prev.landmark[i].x
            dy = curr.landmark[i].y - prev.landmark[i].y

            total += dx*dx + dy*dy
            count += 1

        return total / count if count > 0 else 0

    @staticmethod
    def interpolate(prev_hands, curr_hands):
        if not prev_hands:
            return curr_hands
        if not curr_hands:
            return prev_hands

        matched = []
        used_prev = set()

        for curr in curr_hands:
            cx, cy = HandInterpolator.get_center(curr)

            best_idx = -1
            best_dist = float("inf")

            for i, prev in enumerate(prev_hands):
                if i in used_prev:
                    continue

                px, py = HandInterpolator.get_center(prev)
                dist = (cx - px)**2 + (cy - py)**2

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # 🔥 threshold prevents bad matches
            if best_idx != -1 and best_dist < 0.02:
                used_prev.add(best_idx)
                prev = prev_hands[best_idx]

                velocity = HandInterpolator.get_velocity(prev, curr)

                # 🔥 ADAPTIVE SMOOTHING
                if velocity < 0.0005:
                    alpha = 0.85   # very smooth (slow motion)
                elif velocity < 0.002:
                    alpha = 0.6
                else:
                    alpha = 0.2   # responsive (fast motion)

                new_hand = copy.deepcopy(curr)

                for i in range(len(new_hand.landmark)):
                    new_hand.landmark[i].x = (
                        alpha * prev.landmark[i].x +
                        (1 - alpha) * curr.landmark[i].x
                    )
                    new_hand.landmark[i].y = (
                        alpha * prev.landmark[i].y +
                        (1 - alpha) * curr.landmark[i].y
                    )

                matched.append(new_hand)

            else:
                matched.append(copy.deepcopy(curr))

        return matched