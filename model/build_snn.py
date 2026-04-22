from superneuromat import SNN
import typer
import pandas as pd
from tqdm import tqdm


def build_snn_from_edgelist(edgelist_path: str) -> SNN:
    # Load the edgelist
    edgelist = pd.read_csv(edgelist_path)
    nodelist = pd.unique(edgelist[["source", "target"]].values.ravel())

    # Verify the edgelist properly loaded
    print(f"Number of nodes: {len(nodelist)}")
    
    # Build the SNN using the edgelist
    snn = SNN()

    # Create neurons for each unique node in the edgelist
    neurons = {}
    for node in tqdm(nodelist):
        neuron = snn.create_neuron(threshold=0.9, leak=0.0)
        neurons[node] = neuron
    
    # Create synapses based on the edgelist
    for _, row in tqdm(edgelist.iterrows()):
        source = row["source"]
        target = row["target"]
        snn.create_synapse(neurons[source], neurons[target], weight=1.0, delay=1.0)

    return snn, neurons, nodelist

def all_to_all(net_neurons, feature_neurons, snn):
    for feature in feature_neurons.values():
        for neuron in net_neurons.values():
            snn.create_synapse(feature, neuron, weight=0.5, delay=1.0)

def save_checkpoint(snn: SNN, neurons: dict, feature_neurons: dict, path: str):
    labels = {
        "net": {str(node): neuron.idx for node, neuron in neurons.items()},
        "features": {name: neuron.idx for name, neuron in feature_neurons.items()},
    }
    with open(path, "w") as f:
        snn.saveas_json(f, extra=labels)


def load_checkpoint(path: str) -> tuple[SNN, dict, dict]:
    with open(path) as f:
        raw = f.read()
    import json as _json
    labels = _json.loads(raw).get("extra", {})
    snn = SNN()
    snn.from_jsons(raw)
    net_neurons = {int(node): snn.neurons[idx] for node, idx in labels.get("net", {}).items()}
    feature_neurons = {name: snn.neurons[idx] for name, idx in labels.get("features", {}).items()}
    return snn, net_neurons, feature_neurons


def main(
    edgelist_path: str = typer.Option(
        ...,
        "--edgelist-path",
        help="Path to the edgelist file (CSV with two columns: source, target).",
    ),
    num_features: int = typer.Option(
        16,
        "--num-features",
        help="Number of features for each neuron.",
    ),
    checkpoint_path: str = typer.Option(
        "checkpoint.snn.json",
        "--checkpoint-path",
        help="Path to save the SNN checkpoint.",
    ),
):
    snn, neurons, nodelist = build_snn_from_edgelist(edgelist_path)

    feature_neurons = {}
    for i in range(num_features):
        feature_neuron = snn.create_neuron(threshold=0.9, leak=0.0)
        feature_neurons[f"feature_{i}"] = feature_neuron

    all_to_all(neurons, feature_neurons, snn)

    save_checkpoint(snn, neurons, feature_neurons, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    typer.run(main)
