import torch
from torch.utils.data import DataLoader
from training.dataset_loader import GestureDataset
from training.model import GestureModel


def main():
    dataset = GestureDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GestureModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0

        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "training/model.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()