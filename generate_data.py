"""Generate and save distributed (non-IID) FashionMNIST datasets for each client.

Run this once before starting the federated learning simulation:
    python generate_data.py
"""

import tomli

from pytorchexample.data import generate_distributed_datasets


def main():
    # Read config from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        config = tomli.load(f)

    app_config = config["tool"]["flwr"]["app"]["config"]

    num_clients = int(app_config["num-clients"])
    alpha = float(app_config["alpha-dirichlet"])
    data_dir = app_config["data-dir"]

    print(f"Generating data for {num_clients} clients (alpha={alpha}) -> {data_dir}")
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
