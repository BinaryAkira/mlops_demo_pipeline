"""Project entrypoint."""

import logging

from src.pipeline import run_pipeline


def main() -> None:
    """Run the full ML pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model, acc = run_pipeline()
    print(f"Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
