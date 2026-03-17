from utils.geometry import get_centroid, euclidean

def match_ids_to_boxes(objects, boxes):
    id_to_box = {}

    for obj_id, centroid in objects.items():
        closest_box = None
        min_dist = float("inf")

        for box in boxes:
            box_centroid = get_centroid(box)
            dist = euclidean(centroid, box_centroid)

            if dist < min_dist:
                min_dist = dist
                closest_box = box

        if closest_box is not None:
            id_to_box[obj_id] = closest_box

    return id_to_box