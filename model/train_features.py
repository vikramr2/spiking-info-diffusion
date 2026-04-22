import sys
from pathlib import Path

import numpy as np
import torch
import typer

from build_snn import load_checkpoint

# saelens.py lives in saes/model/
sys.path.insert(0, str(Path(__file__).parent.parent / "saes" / "model"))
from saelens import TopKSAE


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    return x


def load_sparse_embeddings(
    sae_checkpoint: str,
    embeddings_path: str,
    embedding_ids_path: str,
    net_neurons: dict,
) -> dict:
    raw = np.load(embeddings_path).astype(np.float32)
    x = _normalize(torch.from_numpy(raw))

    state = torch.load(sae_checkpoint, map_location="cpu")
    cfg = state["cfg"]
    model = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"])
    model.load_state_dict(state["state_dict"])
    model.eval()

    with torch.no_grad():
        acts, _ = model.encode(x)
    acts = acts.numpy()  # (num_papers, dict_size)

    embedding_ids = np.load(embedding_ids_path).tolist()
    id_to_idx = {pid: i for i, pid in enumerate(embedding_ids)}

    return {node_id: acts[id_to_idx[node_id]]
            for node_id in net_neurons
            if node_id in id_to_idx}


def main(
    checkpoint_path: str = typer.Option(
        ...,
        "--checkpoint-path",
        help="Path to the SNN checkpoint JSON file.",
    ),
    sae_vector_path: str = typer.Option(
        ...,
        "--sae-vector-path",
        help="Path to the SAE checkpoint (.pt file).",
    ),
    embeddings_path: str = typer.Option(
        None,
        "--embeddings-path",
        help="Path to embeddings.npy. Defaults to embeddings.npy in the SAE checkpoint directory.",
    ),
    embedding_ids_path: str = typer.Option(
        None,
        "--embedding-ids-path",
        help="Path to embedding_ids.npy. Defaults to embedding_ids.npy in the SAE checkpoint directory.",
    ),
):
    sae_dir = Path(sae_vector_path).parent
    embeddings_path = embeddings_path or str(sae_dir / "embeddings.npy")
    embedding_ids_path = embedding_ids_path or str(sae_dir / "embedding_ids.npy")

    snn, net_neurons, feature_neurons = load_checkpoint(checkpoint_path)

    print(f"SNN neurons:     {snn.num_neurons}")
    print(f"SNN synapses:    {snn.num_synapses}")
    print(f"Net neurons:     {len(net_neurons)}")
    print(f"Feature neurons: {len(feature_neurons)}")
    assert len(net_neurons) > 0, "No net neurons loaded"
    assert len(feature_neurons) > 0, "No feature neurons loaded"
    assert snn.num_neurons == len(net_neurons) + len(feature_neurons), "Neuron count mismatch"
    print("Checkpoint loaded successfully.")

    print("Loading sparse embeddings...")
    node_embeddings = load_sparse_embeddings(
        sae_vector_path, embeddings_path, embedding_ids_path, net_neurons
    )
    missing = len(net_neurons) - len(node_embeddings)
    print(f"Embeddings aligned: {len(node_embeddings)} / {len(net_neurons)} net neurons ({missing} missing)")
    assert len(node_embeddings) > 0, "No embeddings aligned to net neurons"

    sample_id, sample_vec = next(iter(node_embeddings.items()))
    print(f"Sample node {sample_id}: activation shape {sample_vec.shape}, "
          f"nnz={int((sample_vec > 0).sum())}")


if __name__ == "__main__":
    typer.run(main)
