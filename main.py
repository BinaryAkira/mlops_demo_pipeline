"""Project entrypoint."""

import logging

from src.config.config_loader import load_config
from src.pipeline import run_pipeline


CONFIG_PATH = "config/config.yaml"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(CONFIG_PATH)
    model, acc = run_pipeline(config)

    print(f"Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
