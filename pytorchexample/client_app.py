"""pytorchexample: Flower ClientApp for FashionMNIST federated learning."""

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Array

from pytorchexample.data import load_client_data
from pytorchexample.model import CustomFashionModel

app = ClientApp()


# Helper functions to pack/unpack model parameters

def pack_params(params):
    """Convert a list of numpy arrays into an ArrayRecord (one Array per param)."""
    return ArrayRecord({
        str(i): Array(
            dtype=str(p.dtype),
            shape=tuple(p.shape),  # shape of the tensor (must be tuple)
            stype="numpy.ndarray", # tells flwr how the bytes are encoded
            data=p.tobytes(),      # raw bytes of the array
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


# Train handler

@app.train()
def train(msg: Message, context: Context):
    """Receive global weights, train locally, send back updated weights."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Which data partition this client node owns
    partition_id = context.node_config["partition-id"]

    # Read hyper-parameters from the run config (set in pyproject.toml)
    local_epochs = int(context.run_config["local-epochs"])
    batch_size = int(context.run_config["batch-size"])
    data_dir = context.run_config["data-dir"]
    lr = float(context.run_config["learning-rate"])
    client_optimizer = context.run_config["client-optimizer"]   # "sgd" or "adam"
    client_algorithm = context.run_config["client-algorithm"]   # "fedavg" or "fedsgd"

    # Build the model and load the global weights sent by the server
    model = CustomFashionModel().to(device)
    params = unpack_params(msg.content["arrays"])
    model.set_model_parameters(params)

    # Load this client's local train/val data
    train_loader, _ = load_client_data(partition_id, data_dir, batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    # Select optimizer
    if client_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train: one gradient step (FedSGD) or multiple full epochs (FedAvg)
    if client_algorithm == "fedsgd":
        train_loss, train_acc = model.train_one_step(train_loader, criterion, optimizer, device)
    else:
        for _ in range(local_epochs):
            train_loss, train_acc = model.train_epoch(train_loader, criterion, optimizer, device)

    # Pack updated weights and training metrics into the reply
    arrays_record = pack_params(model.get_model_parameters())
    metrics_record = MetricRecord({
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "num_examples": len(train_loader.dataset),
    })
    content = RecordDict({"arrays": arrays_record, "metrics": metrics_record})
    return Message(content=content, reply_to=msg)


# Evaluate handler

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Receive global weights, evaluate on local validation data, report metrics."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    partition_id = context.node_config["partition-id"]
    batch_size = int(context.run_config["batch-size"])
    data_dir = context.run_config["data-dir"]

    # Build the model and load the global weights sent by the server
    model = CustomFashionModel().to(device)
    params = unpack_params(msg.content["arrays"])
    model.set_model_parameters(params)

    # Load validation data for this client
    _, val_loader = load_client_data(partition_id, data_dir, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    eval_loss, eval_acc = model.test_epoch(val_loader, criterion, device)

    # Pack evaluation metrics into the reply
    metrics_record = MetricRecord({
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num_examples": len(val_loader.dataset),
    })
    content = RecordDict({"metrics": metrics_record})
    return Message(content=content, reply_to=msg)
