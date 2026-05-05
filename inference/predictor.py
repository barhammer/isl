import torch


# ----------------------------------------
# 🔹 Core inference
# ----------------------------------------
def run_static(model, vec, device):
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)

    return probs


# ----------------------------------------
# 🔥 DEBUG VERSION (USE THIS FIRST)
# ----------------------------------------
def debug_static(model, vec, device, label_map, topk=3):
    probs = run_static(model, vec, device)[0]

    values, indices = torch.topk(probs, topk)

    results = []
    print("\n🔍 TOP PREDICTIONS:")

    for i in range(topk):
        idx = indices[i].item()
        conf = values[i].item()
        label = label_map.get(idx, "UNKNOWN")

        print(f"{i+1}. {label} → {conf:.4f}")
        results.append((label, conf))

    print("----")

    return results


# ----------------------------------------
# 🔹 CLEAN VERSION (NO FILTERING)
# ----------------------------------------
def run_static_raw(model, vec, device, label_map):
    probs = run_static(model, vec, device)[0]

    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()
    label_name = label_map.get(pred, "UNKNOWN")

    return pred, confidence, label_name


# ----------------------------------------
# 🔹 WITH MARGIN (USE AFTER DEBUG)
# ----------------------------------------
def run_static_with_margin(
    model,
    vec,
    device,
    label_map,
    none_class,
    margin_threshold=0.1
):
    probs = run_static(model, vec, device)[0]

    sorted_probs, indices = torch.sort(probs, descending=True)

    top1 = sorted_probs[0].item()
    top2 = sorted_probs[1].item()
    pred = indices[0].item()

    margin = top1 - top2

    label_name = label_map.get(pred, "UNKNOWN")

    # 🔥 Debug print (keep this for now)
    print(f"[RAW] {label_name} | conf={top1:.3f} | margin={margin:.3f}")

    if margin < margin_threshold:
        pred = none_class if none_class is not None else pred
        confidence = top1
        label_name = "NONE"
        accepted = False
    else:
        confidence = top1
        accepted = True

    return pred, confidence, label_name, margin, accepted