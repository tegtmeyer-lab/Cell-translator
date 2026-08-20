"""Train scGPT and generate perturbation predictions on the common benchmark cells.

This runner implements the final scGPT benchmark workflow used for the
cross-method comparison:

1. Build/load the GEARS perturbation graph representation from the processed
   neuronal AnnData object.
2. Initialize ``TransformerGenerator`` from the pretrained scGPT brain model.
3. Load the gene encoder, value encoder, and transformer encoder weights.
4. Freeze those pretrained backbone components and fine-tune the remaining
   parameters for two epochs.
5. Match the common CellOT held-out control cells in their original order.
6. Apply each single-gene perturbation mask to those control cells.
7. Predict all genes in sequential chunks and export one raw ``.npz`` file
   per perturbation for downstream benchmark postprocessing.

The downstream conversion to the absolute-expression representation used in
the comparison is handled separately by ``scgpt_postprocess.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from gears import PertData
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id, set_seed
from torch_geometric.loader import DataLoader

from .common import core_cell_id, load_yaml, read_id_file


DEFAULT_BACKBONE_PREFIXES = (
    "encoder",
    "value_encoder",
    "transformer_encoder",
)


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _safe_name(value: str) -> str:
    return str(value).replace("+", "_plus_").replace("/", "_")


def _graph_condition_to_adata_condition(
    graph_condition: str,
    available_conditions: set[str],
    control_label: str,
) -> str:
    """Map a GEARS graph condition back to the matching AnnData condition."""
    if graph_condition == "ctrl":
        if control_label in available_conditions:
            return control_label
        return "ctrl"

    if graph_condition in available_conditions:
        return graph_condition

    if graph_condition.endswith("+ctrl"):
        gene = graph_condition[: -len("+ctrl")]
        if gene in available_conditions:
            return gene

    raise KeyError(
        f"Could not map GEARS condition '{graph_condition}' to an AnnData "
        f"condition. Available conditions include: "
        f"{sorted(available_conditions)[:10]}"
    )


def _attach_obs_names_to_graphs(
    graphs_by_condition: dict,
    adata: ad.AnnData,
    *,
    condition_key: str,
    control_label: str,
) -> None:
    """Attach AnnData observation names to GEARS graphs in condition order.

    This follows the graph/AnnData ordering used by the benchmark workflow.
    """
    available = set(adata.obs[condition_key].astype(str).unique())

    for graph_condition, graphs in graphs_by_condition.items():
        try:
            adata_condition = _graph_condition_to_adata_condition(
                str(graph_condition),
                available,
                control_label,
            )
        except KeyError:
            # Ignore GEARS-generated conditions outside the six benchmark states.
            continue

        mask = adata.obs[condition_key].astype(str).eq(adata_condition)
        obs_names = adata.obs_names[mask].astype(str).tolist()

        if len(obs_names) != len(graphs):
            raise ValueError(
                f"Condition '{graph_condition}' maps to '{adata_condition}', "
                f"but AnnData has {len(obs_names)} cells and GEARS has "
                f"{len(graphs)} graphs."
            )

        for graph, obs_name in zip(graphs, obs_names):
            graph.obs_name = obs_name


def _match_common_control_graphs(
    graphs_by_condition: dict,
    common_ids: list[str],
) -> list:
    """Return control graphs in the exact order of the common CellOT IDs."""
    if "ctrl" not in graphs_by_condition:
        raise KeyError("GEARS graph dictionary does not contain a 'ctrl' condition.")

    core_to_graph = {}
    duplicates = []
    for graph in graphs_by_condition["ctrl"]:
        cid = core_cell_id(getattr(graph, "obs_name", ""))
        if cid in core_to_graph:
            duplicates.append(cid)
        else:
            core_to_graph[cid] = graph

    if duplicates:
        raise ValueError(
            f"Duplicate normalized control-cell IDs were found; examples: "
            f"{duplicates[:5]}"
        )

    matched = []
    missing = []
    for full_id in common_ids:
        graph = core_to_graph.get(core_cell_id(full_id))
        if graph is None:
            missing.append(full_id)
        else:
            matched.append(graph)

    if missing:
        raise ValueError(
            f"Could not match {len(missing)} common CellOT control IDs to "
            f"GEARS control graphs; examples: {missing[:5]}"
        )

    if len(matched) != len(common_ids):
        raise RuntimeError("Common-cell matching did not preserve the expected size.")

    return matched


def _build_training_loader(
    graphs_by_condition: dict,
    conditions: list[str],
    *,
    split_seed: int,
    test_fraction: float,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Create the condition-wise 90:10 training split used for scGPT fine-tuning."""
    rng = np.random.default_rng(split_seed)
    train_graphs = []

    for condition in conditions:
        if condition not in graphs_by_condition:
            raise KeyError(
                f"Condition '{condition}' was not found in the GEARS graph dictionary."
            )

        graphs = graphs_by_condition[condition]
        indices = rng.permutation(len(graphs))
        n_test = int(np.floor(test_fraction * len(graphs)))
        train_idx = indices[n_test:]
        train_graphs.extend(graphs[i] for i in train_idx)

    if len(train_graphs) < batch_size:
        raise ValueError(
            f"Only {len(train_graphs)} training graphs are available for "
            f"batch_size={batch_size}."
        )

    return DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )


def load_scgpt_checkpoint_with_wqkv_conversion(
    model: torch.nn.Module,
    model_file: str | Path,
    *,
    prefixes: tuple[str, ...] = DEFAULT_BACKBONE_PREFIXES,
) -> torch.nn.Module:
    """Load selected pretrained scGPT components, converting fused QKV if needed."""
    checkpoint = torch.load(model_file, map_location="cpu")
    model_state = model.state_dict()

    def compatible(key: str, value: torch.Tensor) -> bool:
        return (
            key in model_state
            and tuple(model_state[key].shape) == tuple(value.shape)
        )

    converted = {}
    copied = 0
    converted_qkv = 0

    for key, value in checkpoint.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue

        if compatible(key, value):
            converted[key] = value
            copied += 1
            continue

        if key.endswith("self_attn.Wqkv.weight"):
            base = key.replace("self_attn.Wqkv.weight", "self_attn.")
            in_proj_key = base + "in_proj_weight"

            if compatible(in_proj_key, value):
                converted[in_proj_key] = value
                converted_qkv += 1
                continue

            q_key = base + "q_proj.weight"
            k_key = base + "k_proj.weight"
            v_key = base + "v_proj.weight"
            split_keys = (q_key, k_key, v_key)

            if all(k in model_state for k in split_keys) and value.shape[0] % 3 == 0:
                width = value.shape[0] // 3
                pieces = (value[:width], value[width : 2 * width], value[2 * width :])
                if all(compatible(k, piece) for k, piece in zip(split_keys, pieces)):
                    converted.update(dict(zip(split_keys, pieces)))
                    converted_qkv += 1
                    continue

        if key.endswith("self_attn.Wqkv.bias"):
            base = key.replace("self_attn.Wqkv.bias", "self_attn.")
            in_proj_key = base + "in_proj_bias"

            if compatible(in_proj_key, value):
                converted[in_proj_key] = value
                converted_qkv += 1
                continue

            q_key = base + "q_proj.bias"
            k_key = base + "k_proj.bias"
            v_key = base + "v_proj.bias"
            split_keys = (q_key, k_key, v_key)

            if all(k in model_state for k in split_keys) and value.shape[0] % 3 == 0:
                width = value.shape[0] // 3
                pieces = (value[:width], value[width : 2 * width], value[2 * width :])
                if all(compatible(k, piece) for k, piece in zip(split_keys, pieces)):
                    converted.update(dict(zip(split_keys, pieces)))
                    converted_qkv += 1

    incompatible = model.load_state_dict(converted, strict=False)

    print(
        "Loaded pretrained scGPT parameters: "
        f"direct={copied}, converted_qkv={converted_qkv}"
    )
    print(
        f"Missing keys={len(incompatible.missing_keys)}; "
        f"unexpected keys={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys:
        print("Example missing keys:", incompatible.missing_keys[:10])

    return model


def _build_model(
    *,
    genes: list[str],
    pretrained_model_dir: str | Path,
    device: torch.device,
    dropout: float,
    use_fast_transformer: bool,
    backbone_prefixes: tuple[str, ...],
) -> tuple[TransformerGenerator, np.ndarray]:
    model_dir = Path(pretrained_model_dir)
    config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    for path in (config_file, model_file, vocab_file):
        if not path.exists():
            raise FileNotFoundError(path)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0
    pert_pad_id = 2

    vocab = GeneVocab.from_file(vocab_file)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[pad_token])

    gene_ids = np.asarray(
        [vocab[g] if g in vocab else vocab[pad_token] for g in genes],
        dtype=int,
    )

    with config_file.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    model = TransformerGenerator(
        len(vocab),
        model_config["embsize"],
        model_config["nheads"],
        model_config["d_hid"],
        model_config["nlayers"],
        nlayers_cls=model_config["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=False,
        cell_emb_style="cls",
        mvc_decoder_style="inner product, detach",
        use_fast_transformer=use_fast_transformer,
    ).to(device)

    model = load_scgpt_checkpoint_with_wqkv_conversion(
        model,
        model_file,
        prefixes=backbone_prefixes,
    )

    for name, parameter in model.named_parameters():
        if name.startswith(backbone_prefixes):
            parameter.requires_grad = False

    trainable = sum(parameter.requires_grad for parameter in model.parameters())
    total = sum(1 for _ in model.parameters())
    print(f"Trainable parameter tensors: {trainable}/{total}")

    return model, gene_ids


def _fine_tune(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    gene_ids: np.ndarray,
    n_genes: int,
    device: torch.device,
    max_seq_len: int,
    epochs: int,
    learning_rate: float,
    gradient_clip: float,
) -> None:
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            batch_size = len(batch.y)
            batch = batch.to(device)

            x = batch.x
            original_values = x[:, 0].view(batch_size, n_genes)
            perturbation_flags = x[:, 1].long().view(batch_size, n_genes)
            targets = batch.y

            gene_index = torch.randperm(n_genes, device=device)[:max_seq_len]

            input_values = original_values[:, gene_index]
            input_flags = perturbation_flags[:, gene_index]
            target_values = targets[:, gene_index]

            mapped_gene_ids = map_raw_id_to_vocab_id(
                gene_index,
                gene_ids,
            ).repeat(batch_size, 1)

            padding_mask = torch.zeros_like(
                input_values,
                dtype=torch.bool,
                device=device,
            )

            output = model(
                mapped_gene_ids,
                input_values,
                input_flags,
                src_key_padding_mask=padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            predicted_values = output["mlm_output"]

            mask = torch.ones_like(
                input_values,
                dtype=torch.bool,
                device=device,
            )
            loss = masked_mse_loss(predicted_values, target_values, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1

        print(
            f"Epoch {epoch}/{epochs} "
            f"train loss={total_loss / max(n_steps, 1):.6f}"
        )


def _make_perturbation_mask(
    perturbation: str,
    *,
    gene_to_index: dict[str, int],
    n_genes: int,
) -> np.ndarray:
    if perturbation not in gene_to_index:
        raise ValueError(f"Perturbation gene '{perturbation}' is absent from the gene list.")

    mask = np.zeros(n_genes, dtype=np.int64)
    mask[gene_to_index[perturbation]] = 1
    return mask


def _predict_batch_all_genes(
    model: torch.nn.Module,
    batch,
    *,
    perturbation_mask: np.ndarray,
    gene_ids: np.ndarray,
    n_genes: int,
    max_seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Predict all genes in sequential chunks for one control-cell batch."""
    batch_size = batch.y.shape[0]
    x = batch.x
    original_values = x[:, 0].view(batch_size, n_genes)

    perturbation_flags = (
        torch.as_tensor(
            perturbation_mask,
            device=device,
            dtype=torch.long,
        )
        .view(1, n_genes)
        .repeat(batch_size, 1)
    )

    predictions = np.zeros((batch_size, n_genes), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_genes, max_seq_len):
            end = min(n_genes, start + max_seq_len)
            gene_index = torch.arange(start, end, device=device)

            input_values = original_values[:, gene_index]
            input_flags = perturbation_flags[:, gene_index]
            mapped_gene_ids = map_raw_id_to_vocab_id(
                gene_index,
                gene_ids,
            ).repeat(batch_size, 1)

            padding_mask = torch.zeros_like(
                input_values,
                dtype=torch.bool,
                device=device,
            )

            output = model(
                mapped_gene_ids,
                input_values,
                input_flags,
                src_key_padding_mask=padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )

            predictions[:, start:end] = (
                output["mlm_output"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return predictions


def _export_common_control_predictions(
    model: torch.nn.Module,
    control_graphs: list,
    common_ids: list[str],
    *,
    perturbation: str,
    genes: list[str],
    gene_ids: np.ndarray,
    n_genes: int,
    max_seq_len: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: str | Path,
) -> Path:
    gene_to_index = {gene: i for i, gene in enumerate(genes)}
    perturbation_mask = _make_perturbation_mask(
        perturbation,
        gene_to_index=gene_to_index,
        n_genes=n_genes,
    )

    loader = DataLoader(
        control_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    predictions = []
    observed_control = []

    for batch in loader:
        batch = batch.to(device)

        predictions.append(
            _predict_batch_all_genes(
                model,
                batch,
                perturbation_mask=perturbation_mask,
                gene_ids=gene_ids,
                n_genes=n_genes,
                max_seq_len=max_seq_len,
                device=device,
            )
        )
        observed_control.append(
            batch.y.detach().cpu().numpy().astype(np.float32)
        )

    yhat = np.vstack(predictions)
    y = np.vstack(observed_control)

    if yhat.shape != (len(common_ids), n_genes):
        raise ValueError(
            f"{perturbation}: predicted shape {yhat.shape}, expected "
            f"{(len(common_ids), n_genes)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perturbation_label = f"{perturbation}+ctrl"
    output_path = (
        output_dir
        / f"{_safe_name(perturbation_label)}_CTRL{len(common_ids)}.npz"
    )

    np.savez_compressed(
        output_path,
        yhat=yhat,
        y=y,
        gene_names=np.asarray(genes, dtype=object),
        pert=perturbation_label,
        obs_names=np.asarray(common_ids, dtype=object),
    )

    print(f"[{perturbation}] saved {output_path} shape={yhat.shape}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Override scgpt.device from the YAML (e.g. cpu, cuda, auto).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    scfg = cfg["scgpt"]
    dcfg = cfg["dataset"]

    seed = int(scfg.get("seed", 42))
    split_seed = int(scfg.get("split_seed", 1))
    test_fraction = float(scfg.get("test_fraction", 0.10))
    train_batch_size = int(scfg.get("train_batch_size", 8))
    inference_batch_size = int(scfg.get("inference_batch_size", 8))
    num_workers = int(scfg.get("num_workers", 0))
    max_seq_len = int(scfg.get("max_seq_len", 256))
    epochs = int(scfg.get("epochs", 2))
    learning_rate = float(scfg.get("learning_rate", 1e-3))
    gradient_clip = float(scfg.get("gradient_clip", 1.0))
    dropout = float(scfg.get("dropout", 0.2))
    use_fast_transformer = bool(scfg.get("use_fast_transformer", False))
    backbone_prefixes = tuple(
        scfg.get("backbone_prefixes", DEFAULT_BACKBONE_PREFIXES)
    )

    requested_device = (
        args.device
        if args.device is not None
        else scfg.get("device", "cpu")
    )
    device = _resolve_device(requested_device)

    set_seed(seed)
    print(f"Device: {device}")
    print(
        "scGPT fine-tuning: "
        f"seed={seed}, split_seed={split_seed}, max_seq_len={max_seq_len}, "
        f"epochs={epochs}, lr={learning_rate}, gradient_clip={gradient_clip}"
    )

    h5ad_path = Path(dcfg["h5ad"])
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    adata = ad.read_h5ad(h5ad_path)
    condition_key = str(dcfg.get("condition_key", "condition"))
    control_label = str(dcfg.get("control_label", "ctrl"))
    gene_name_key = str(scfg.get("gene_name_key", "gene_name"))

    if condition_key not in adata.obs:
        raise KeyError(f"AnnData .obs is missing '{condition_key}'.")

    if gene_name_key not in adata.var:
        # The benchmark data use var_names as gene symbols; expose them using
        # the GEARS/scGPT field expected by the workflow.
        adata.var[gene_name_key] = adata.var_names.astype(str)

    genes = adata.var[gene_name_key].astype(str).tolist()
    n_genes = len(genes)

    expected_genes = int(dcfg.get("expected_shape", [adata.n_obs, n_genes])[1])
    if n_genes != expected_genes:
        raise ValueError(
            f"AnnData has {n_genes} genes; expected {expected_genes}."
        )

    missing_perturbations = [g for g in cfg["perturbations"] if g not in set(genes)]
    if missing_perturbations:
        raise ValueError(
            "Perturbation genes absent from the expression matrix: "
            f"{missing_perturbations}"
        )

    gears_workdir = Path(scfg["gears_workdir"])
    gears_dataset_name = str(
        scfg.get("gears_dataset_name", "neuron_5perts_plusctrl")
    )
    gears_dataset_dir = gears_workdir / gears_dataset_name

    if bool(scfg.get("rebuild_gears_data", True)):
        shutil.rmtree(gears_dataset_dir, ignore_errors=True)

    pert_data = PertData(str(gears_workdir))
    if not gears_dataset_dir.exists():
        pert_data.new_data_process(
            dataset_name=gears_dataset_name,
            adata=adata,
        )
    pert_data.load(data_path=str(gears_dataset_dir))

    graph_file = gears_dataset_dir / "data_pyg" / "cell_graphs.pkl"
    if not graph_file.exists():
        raise FileNotFoundError(graph_file)

    with graph_file.open("rb") as handle:
        graphs_by_condition = pickle.load(handle)

    _attach_obs_names_to_graphs(
        graphs_by_condition,
        adata,
        condition_key=condition_key,
        control_label=control_label,
    )

    common_ids = read_id_file(cfg["common_test_control_ids"])
    expected_common = int(dcfg.get("expected_test_control_cells", len(common_ids)))
    if len(common_ids) != expected_common:
        raise ValueError(
            f"Common-ID file contains {len(common_ids)} cells; expected "
            f"{expected_common}."
        )

    common_control_graphs = _match_common_control_graphs(
        graphs_by_condition,
        common_ids,
    )
    print(f"Matched common control graphs: {len(common_control_graphs)}")

    training_conditions = ["ctrl"] + [
        f"{gene}+ctrl" for gene in cfg["perturbations"]
    ]
    train_loader = _build_training_loader(
        graphs_by_condition,
        training_conditions,
        split_seed=split_seed,
        test_fraction=test_fraction,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )

    model, gene_ids = _build_model(
        genes=genes,
        pretrained_model_dir=scfg["pretrained_model_dir"],
        device=device,
        dropout=dropout,
        use_fast_transformer=use_fast_transformer,
        backbone_prefixes=backbone_prefixes,
    )

    _fine_tune(
        model,
        train_loader,
        gene_ids=gene_ids,
        n_genes=n_genes,
        device=device,
        max_seq_len=max_seq_len,
        epochs=epochs,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
    )
    print("scGPT head fine-tuning complete.")

    raw_output_dir = Path(
        scfg.get(
            "raw_output_dir",
            Path(scfg["output_root"]) / "raw",
        )
    )

    for perturbation in cfg["perturbations"]:
        _export_common_control_predictions(
            model,
            common_control_graphs,
            common_ids,
            perturbation=perturbation,
            genes=genes,
            gene_ids=gene_ids,
            n_genes=n_genes,
            max_seq_len=max_seq_len,
            device=device,
            batch_size=inference_batch_size,
            num_workers=num_workers,
            output_dir=raw_output_dir,
        )


if __name__ == "__main__":
    main()
