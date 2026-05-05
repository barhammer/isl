# 🧠 Multi-Person ISL Recognition System

Real-time Indian Sign Language (ISL) recognition using MediaPipe + PyTorch.

🚀 **Core Innovation:**
Unlike traditional MediaPipe-based systems that assume a single signer, this system supports
👉 **multiple people signing simultaneously**, each with independent tracking and prediction.

---

## 🔗 Repository

GitHub: https://github.com/barhammer/isl

Clone:

```bash
git clone https://github.com/barhammer/isl.git
cd isl
```

---

# ⚡ One-Command Setup (RECOMMENDED)

👉 This is the easiest and safest way to run the project.

```bash
chmod +x setup.sh
./setup.sh
conda activate isl
python main.py
```

---

# 🔧 Manual Setup (Alternative)

```bash
conda create -n isl python=3.10 -y
conda activate isl

pip install torch torchvision
pip install opencv-python mediapipe numpy
```

---

# ▶️ Run the system

```bash
python main.py
```

Press **ESC** to exit.

---

# 🧠 How the system works

1. **MediaPipe** → detects hands & landmarks
2. **Pipeline** → assigns IDs to multiple people
3. **Model** → predicts gesture per person
4. **Output** → label displayed next to each person

👉 Each person is tracked independently in real-time.

---

# 🔥 Core Feature (Demo Highlight)

✅ Supports **multiple people signing at the same time**
✅ Each person gets:

* unique ID
* independent gesture prediction

---

# ⚠️ IMPORTANT NOTES

## 1. GPU is OPTIONAL

System automatically uses:

* GPU (if available)
* CPU (fallback)

If GPU causes issues:

```bash
CUDA_VISIBLE_DEVICES="" python main.py
```

---

## 2. Requires Conda

Install Miniconda or Anaconda before running setup script.

---

## 3. Camera Issues

Check available devices:

```bash
ls /dev/video*
```

Then modify in `main.py`:

```python
camera = Camera(0)  # try 1 or 2 if needed
```

---

# 🛠️ Common Issues + Fixes

---

### ❌ Import errors

```bash
pip install <missing-package>
```

---

### ❌ Model not loading

Check file:

```
training/static_model.pth
```

If missing:

```bash
python -m training.train_static
```

---

### ❌ Wrong predictions

Make sure:

* same distance as training
* hand centered
* hold gesture ~1 second

---

### ❌ Low FPS

Quick fix:

```python
delay = 50
```

Close background apps if needed.

---

### ❌ Camera not opening

Try:

```python
camera = Camera(1)
```

---

# 🧪 Training (Optional)

```bash
python -m training.train_static
```

---

# 📁 Project Structure

```
training/        → dataset + model training
inference/       → model loading
pipeline/        → mediapipe processing
runtime/         → FPS utilities
main.py          → entry point
setup.sh         → one-command installer
```

---

# 🔥 Demo Tips (VERY IMPORTANT)

* Use **clear, distinct gestures**
* Hold gesture briefly (0.5–1 sec)
* Stay at consistent distance
* Avoid rapid switching

👉 System is optimized for **static gestures**

---

# ✅ Pre-Demo Checklist

* [ ] Setup script runs successfully
* [ ] Camera works
* [ ] Model loads
* [ ] Tested multiple people simultaneously
* [ ] FPS acceptable

---

# 🚀 If something breaks

1. Run on CPU
2. Restart program
3. Keep gestures simple
4. Re-run

👉 System is designed to degrade gracefully, not crash.

---

# 💡 Notes

* Dynamic / LSTM code exists but is disabled for stability
* System currently optimized for static recognition
* Focus is real-time performance and robustness

---

# 🎯 Final Goal

This project demonstrates:

👉 **Reliable real-time ISL recognition for multiple users simultaneously**

---
