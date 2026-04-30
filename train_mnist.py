import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_file: Path,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        global_step += 1
        batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        avg_loss = total_loss / total
        avg_acc = correct / total
        log_message(
            log_file,
            f"step={global_step} epoch={epoch} iter={batch_idx}/{len(loader)} "
            f"loss={loss.item():.4f} batch_acc={batch_acc:.4f} "
            f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}",
        )

    return total_loss / total, correct / total, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def log_message(log_file: Path, message: str) -> None:
    print(message, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple MNIST MLP.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    checkpoint_path = run_dir / "checkpoint.pt"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_message(log_file, f"device={device}")
    log_message(log_file, f"results_dir={run_dir}")
    global_step = 0
    final_test_loss = 0.0
    final_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            global_step,
            log_file,
        )
        final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
        log_message(
            log_file,
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={final_test_loss:.4f} test_acc={final_test_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs": args.epochs,
            "global_step": global_step,
            "test_loss": final_test_loss,
            "test_acc": final_test_acc,
        },
        checkpoint_path,
    )
    log_message(log_file, f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
