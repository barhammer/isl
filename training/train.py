import os
import torch
from torch.utils.data import DataLoader
from training.dataset_loader import GestureDataset
from training.model import LSTMGestureModel   # 🔥 renamed model


def main():
    # ----------------------------------------
    # 🔹 Device
    # ----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ----------------------------------------
    # 🔹 Dataset (SINGLE SOURCE OF TRUTH)
    # ----------------------------------------
    train_dataset = GestureDataset(split="train")

    if len(train_dataset) == 0:
        print("❌ No training data found.")
        return

    # 🔥 Extract label_map ONCE
    label_map = train_dataset.label_map
    num_classes = len(label_map)

    # 🔥 Force SAME mapping for validation
    val_dataset = GestureDataset(
        split="val",
        label_map=label_map
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ----------------------------------------
    # 🔹 Model (FIXED)
    # ----------------------------------------
    model = LSTMGestureModel(
        input_size=162,
        hidden_size=64,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ----------------------------------------
    # 🔹 LR Scheduler
    # ----------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1
    )

    best_val_loss = float("inf")

    # ----------------------------------------
    # 🔹 Early stopping
    # ----------------------------------------
    early_stop_patience = 2
    min_delta = 0.01
    no_improve = 0

    max_epochs = 20

    # ----------------------------------------
    # 🔁 Training loop
    # ----------------------------------------
    for epoch in range(max_epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0
        train_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = loss_fn(out, y)

                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= max(val_batches, 1)
        val_acc = (correct / total) * 100 if total > 0 else 0

        # ----------------------------------------
        # 🔹 LR Scheduling
        # ----------------------------------------
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr < prev_lr:
            print(f"📉 LR reduced: {prev_lr:.6f} → {new_lr:.6f}")

        # ----------------------------------------
        # 🔹 Print stats
        # ----------------------------------------
        print(
            f"Epoch {epoch+1} | "
            f"LR: {new_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # ----------------------------------------
        # 🔥 Save best model (FIXED)
        # ----------------------------------------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve = 0

            save_path = os.path.join(os.path.dirname(__file__), "lstm_model.pth")

            torch.save({
                "model": model.state_dict(),
                "label_map": label_map   # 🔥 CRITICAL
            }, save_path)

            print(f"💾 Saved best LSTM model → {save_path}")

        else:
            no_improve += 1
            print(f"⚠️ No improvement ({no_improve}/{early_stop_patience})")

        # ----------------------------------------
        # 🔥 Early stopping
        # ----------------------------------------
        if no_improve >= early_stop_patience:
            print("🛑 Early stopping triggered")
            break

    print("✅ LSTM training complete.")


if __name__ == "__main__":
    main()