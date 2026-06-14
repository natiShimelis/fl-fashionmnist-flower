"""pytorchexample: Flower ServerApp for FashionMNIST federated learning."""

import json
import pathlib
import random as stdlib_random

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.common import Array, MessageType
from flwr.serverapp import Grid, ServerApp

from pytorchexample.model import CustomFashionModel

app = ServerApp()


# Helper functions to pack/unpack model parameters

def pack_params(params):
    """Convert a list of numpy arrays into an ArrayRecord."""
    return ArrayRecord({
        str(i): Array(
            dtype=str(p.dtype),
            shape=tuple(p.shape),
            stype="numpy.ndarray",
            data=p.tobytes(),
        )
        for i, p in enumerate(params)
    })


def unpack_params(array_record):
    """Convert an ArrayRecord back into a list of numpy arrays."""
    n = len(array_record)
    return [
        np.frombuffer(array_record[str(i)].data, dtype=array_record[str(i)].dtype)
        .reshape(array_record[str(i)].shape)
        for i in range(n)
    ]


# Aggregation helpers

def sample_clients(node_ids, fraction, rng):
    """Randomly select a fraction of clients for a round."""
    k = max(1, int(len(node_ids) * fraction))
    return rng.sample(list(node_ids), k)


def fedavg(state_dicts, num_examples):
    """Compute a weighted average of state dicts proportional to num_examples."""
    total = sum(num_examples)
    avg = {}
    for key in state_dicts[0]:
        # Weighted sum over all clients for this parameter
        avg[key] = sum(
            state_dicts[i][key].float() * (num_examples[i] / total)
            for i in range(len(state_dicts))
        )
    return avg


def aggregate_metrics(values, num_examples):
    """Compute a weighted average of a scalar metric across clients."""
    total = sum(num_examples)
    return sum(v * n / total for v, n in zip(values, num_examples))


# Main server loop

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the federated learning loop for num_server_rounds rounds."""

    # Read run config from pyproject.toml
    num_server_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])
    lr = float(context.run_config["learning-rate"])
    seed = int(context.run_config["seed"])
    local_epochs = int(context.run_config["local-epochs"])
    batch_size = int(context.run_config["batch-size"])
    data_dir = context.run_config["data-dir"]
    client_optimizer = context.run_config["client-optimizer"]
    client_algorithm = context.run_config["client-algorithm"]

    # Initialize global model and reproducible RNG
    global_model = CustomFashionModel()
    rng = stdlib_random.Random(seed)

    results = []  # will hold one dict per round

    for round_num in range(1, num_server_rounds + 1):
        print(f"\n--- Round {round_num}/{num_server_rounds} ---")

        # 1. Select clients for this round
        all_node_ids = list(grid.get_node_ids())
        selected_ids = sample_clients(all_node_ids, fraction_train, rng)
        print(f"Selected {len(selected_ids)} clients: {selected_ids}")

        # 2. Pack current global model weights
        global_params = global_model.get_model_parameters()
        arrays_record = pack_params(global_params)

        # 3. Send train messages and collect replies
        train_content = RecordDict({"arrays": arrays_record})
        train_msgs = [
            grid.create_message(
                content=train_content,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=f"round-{round_num}-train",
            )
            for node_id in selected_ids
        ]
        train_replies = list(grid.send_and_receive(train_msgs))

        # Collect state dicts and metrics from successful replies
        client_state_dicts = []
        client_train_losses = []
        client_train_accs = []
        client_train_examples = []

        for reply in train_replies:
            if reply.has_error():
                print(f"  Node returned an error, skipping.")
                continue
            params = unpack_params(reply.content["arrays"])
            # Convert numpy params → torch tensors keyed by layer name
            keys = list(global_model.state_dict().keys())
            state_dict = {k: torch.tensor(p) for k, p in zip(keys, params)}
            client_state_dicts.append(state_dict)

            metrics = reply.content["metrics"]
            client_train_losses.append(float(metrics["train_loss"]))
            client_train_accs.append(float(metrics["train_acc"]))
            client_train_examples.append(int(metrics["num_examples"]))

        if not client_state_dicts:
            print("  No successful train replies, skipping round.")
            continue

        # 4. FedAvg: aggregate client weights into new global model
        new_state_dict = fedavg(client_state_dicts, client_train_examples)
        global_model.load_state_dict(new_state_dict)

        # Weighted-average train metrics
        avg_train_loss = aggregate_metrics(client_train_losses, client_train_examples)
        avg_train_acc = aggregate_metrics(client_train_accs, client_train_examples)

        # 5. Send evaluate messages with the updated global weights
        eval_arrays = pack_params(global_model.get_model_parameters())
        eval_content = RecordDict({"arrays": eval_arrays})
        eval_msgs = [
            grid.create_message(
                content=eval_content,
                message_type=MessageType.EVALUATE,
                dst_node_id=node_id,
                group_id=f"round-{round_num}-eval",
            )
            for node_id in selected_ids
        ]
        eval_replies = list(grid.send_and_receive(eval_msgs))

        client_eval_losses = []
        client_eval_accs = []
        client_eval_examples = []

        for reply in eval_replies:
            if reply.has_error():
                continue
            metrics = reply.content["metrics"]
            client_eval_losses.append(float(metrics["eval_loss"]))
            client_eval_accs.append(float(metrics["eval_acc"]))
            client_eval_examples.append(int(metrics["num_examples"]))

        # Weighted-average eval metrics (fall back to 0 if no replies)
        if client_eval_examples:
            avg_eval_loss = aggregate_metrics(client_eval_losses, client_eval_examples)
            avg_eval_acc = aggregate_metrics(client_eval_accs, client_eval_examples)
        else:
            avg_eval_loss, avg_eval_acc = 0.0, 0.0

        # 6. Print summary and record results
        print(
            f"  train_loss={avg_train_loss:.4f}  train_acc={avg_train_acc:.4f}"
            f"  eval_loss={avg_eval_loss:.4f}  eval_acc={avg_eval_acc:.4f}"
        )

        results.append({
            "round": round_num,
            "num_clients": len(client_state_dicts),
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "eval_loss": avg_eval_loss,
            "eval_acc": avg_eval_acc,
        })

    # 7. Save results to disk
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{context.run_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
