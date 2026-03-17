def extract_person_boxes(results):
    boxes = []

    r = results[0]

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != 0 or conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))

    return boxes