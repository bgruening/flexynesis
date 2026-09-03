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
