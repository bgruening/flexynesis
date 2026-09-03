"""Edge weights on the GNN graph."""

import pandas as pd
import pytest
import torch

from flexynesis.data import MultiOmicDatasetNW
from flexynesis.modules import flexGCN


def build_edges(pairs_and_scores, genes=("A", "B", "C")):
    """Run create_edge_index against a hand-built interaction table."""
    dataset = object.__new__(MultiOmicDatasetNW)
    dataset.interaction_df = pd.DataFrame(
        [(a, b, s) for a, b, s in pairs_and_scores],
        columns=["protein1", "protein2", "combined_score"],
    )
    dataset.common_features = list(genes)
    dataset.gene_to_index = {g: i for i, g in enumerate(genes)}
    return dataset.create_edge_index()


def test_weights_align_with_edges():
    edge_index, edge_weight = build_edges([("A", "B", 1.0), ("B", "C", 0.25)])
    assert edge_index.shape == (2, 2)
    assert edge_index.dtype == torch.long
    assert edge_weight.tolist() == pytest.approx([1.0, 0.25])


def test_zero_weight_edge_is_kept():
    """A zero weight means "no connection" but must keep the node in the graph."""
    edge_index, edge_weight = build_edges([("A", "B", 0.0), ("B", "C", 0.5)])
    assert edge_index.shape == (2, 2)
    assert edge_weight.tolist() == pytest.approx([0.0, 0.5])


def test_scores_above_one_are_normalised():
    """STRING's native 0-1000 combined scores rescale without caller action."""
    _, edge_weight = build_edges([("A", "B", 1000.0), ("B", "C", 400.0)])
    assert edge_weight.max().item() == pytest.approx(1.0)
    assert edge_weight.tolist() == pytest.approx([1.0, 0.4])


def test_negative_scores_are_shifted_not_clipped():
    """Anti-correlations keep their ordering instead of collapsing onto 0."""
    _, edge_weight = build_edges([("A", "B", -1.0), ("B", "C", 1.0)])
    # -1 -> 0, +1 -> 1; a zero correlation would land midway.
    assert edge_weight.tolist() == pytest.approx([0.0, 1.0])

    _, edge_weight = build_edges(
        [("A", "B", -1.0), ("B", "C", 0.0)], genes=("A", "B", "C")
    )
    assert edge_weight.tolist() == pytest.approx([0.0, 1.0])


def test_edges_outside_the_feature_set_are_dropped():
    edge_index, edge_weight = build_edges(
        [("A", "B", 1.0), ("A", "Z", 1.0)], genes=("A", "B")
    )
    assert edge_index.shape == (2, 1)
    assert edge_weight.tolist() == pytest.approx([1.0])


@pytest.mark.parametrize("conv,supported", [("GCN", True), ("GC", True),
                                            ("SAGE", False)])
def test_conv_edge_weight_support(conv, supported):
    model = flexGCN(
        node_count=3, node_feature_count=1, node_embedding_dim=4,
        output_dim=2, conv=conv,
    )
    assert model.supports_edge_weight is supported

    x = torch.randn(2, 3, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_weight = torch.tensor([1.0, 0.0])

    if supported:
        assert model(x, edge_index, edge_weight).shape == (2, 2)
    else:
        # SAGEConv has no slot for a weight; it must warn rather than crash.
        with pytest.warns(UserWarning, match="ignores"):
            assert model(x, edge_index, edge_weight).shape == (2, 2)


def test_weights_change_the_output():
    """Guard against the weights being accepted and then silently ignored."""
    torch.manual_seed(0)
    model = flexGCN(
        node_count=3, node_feature_count=1, node_embedding_dim=4,
        output_dim=2, conv="GCN", dropout_rate=0.0,
    ).eval()
    x = torch.randn(2, 3, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])

    full = model(x, edge_index, torch.tensor([1.0, 1.0]))
    none = model(x, edge_index, torch.tensor([0.0, 0.0]))
    assert not torch.allclose(full, none)


READOUTS = ["flatten", "mean", "sum", "max", "meanmax"]


@pytest.mark.parametrize("readout", READOUTS)
def test_readout_shapes(readout):
    model = flexGCN(
        node_count=5, node_feature_count=1, node_embedding_dim=4,
        output_dim=3, conv="GCN", readout=readout,
    )
    x = torch.randn(2, 5, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert model(x, edge_index).shape == (2, 3)


def test_unknown_readout_rejected():
    with pytest.raises(ValueError):
        flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                output_dim=3, readout="nonsense")


def test_pooling_readout_is_permutation_invariant():
    """The point of pooling: node order stops carrying information.

    With flatten, relabelling the nodes changes the output, so the model can
    read a node directly and the graph is optional. With mean it cannot.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 5, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])

    # Relabel the nodes: the graph is the same, only the ordering changes, so
    # the edges have to be remapped too or this is a different graph.
    perm = torch.tensor([4, 3, 2, 1, 0])
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))
    x_perm, edges_perm = x[:, perm], inverse[edge_index]

    pooled = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                     output_dim=3, conv="GCN", readout="mean",
                     dropout_rate=0.0).eval()
    flat = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                   output_dim=3, conv="GCN", readout="flatten",
                   dropout_rate=0.0).eval()

    assert torch.allclose(pooled(x, edge_index),
                          pooled(x_perm, edges_perm), atol=1e-5)
    assert not torch.allclose(flat(x, edge_index), flat(x_perm, edges_perm))


def test_pooling_shrinks_the_readout():
    """Most of the model should stop living in the final layer."""
    def readout_share(readout):
        model = flexGCN(
            node_count=48, node_feature_count=1, node_embedding_dim=9,
            output_dim=105, num_convs=3, conv="GCN", readout=readout,
        )
        total = sum(p.numel() for p in model.parameters())
        return sum(p.numel() for p in model.fc.parameters()) / total

    assert readout_share("flatten") > 0.98
    assert readout_share("mean") < 0.85
