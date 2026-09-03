"""Edge weights on the GNN graph."""

from itertools import combinations

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


def test_zero_score_is_not_an_edge():
    """0 means "no connection", so it must not survive as an edge.

    Otherwise a network listing every gene pair reads as fully connected the
    moment edge weighting is turned off.
    """
    edge_index, edge_weight = build_edges([("A", "B", 0.0), ("B", "C", 0.5)])
    assert edge_index.shape == (2, 1)
    assert edge_weight.tolist() == pytest.approx([0.5])


def test_zero_score_row_still_keeps_the_gene_as_a_node():
    """The all-pairs listing keeps unwired genes in the graph."""
    import pandas as pd

    dataset = object.__new__(MultiOmicDatasetNW)
    dataset.multiomic_dataset = type("D", (), {"features": {"m": ["A", "B", "C"]}})()
    dataset.interaction_df = pd.DataFrame(
        [("A", "B", 0.5), ("A", "C", 0.0), ("B", "C", 0.0)],
        columns=["protein1", "protein2", "combined_score"],
    )
    # C appears only in zero-score rows, and must still be a node
    assert "C" in dataset.find_union_features()


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

    # A middle value lands midway once the range is shifted and rescaled.
    _, edge_weight = build_edges(
        [("A", "B", -1.0), ("B", "C", 0.5), ("A", "C", 1.0)],
        genes=("A", "B", "C"),
    )
    assert edge_weight.tolist() == pytest.approx([0.0, 0.75, 1.0])


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


READOUTS = ["mean", "sum", "max", "meanmax", "attention", "dim_attention"]


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

    A readout that kept a weight per node could read a node directly and treat
    the graph as optional. dim_attention keeps gene identity deliberately; mean
    does not.
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
    keyed = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="dim_attention",
                    project=False, dropout_rate=0.0).eval()

    assert torch.allclose(pooled(x, edge_index),
                          pooled(x_perm, edges_perm), atol=1e-5)
    assert not torch.allclose(keyed(x, edge_index), keyed(x_perm, edges_perm))


def test_pooling_shrinks_the_readout():
    """Most of the model should not live in the final layer."""
    def readout_share(readout):
        model = flexGCN(
            node_count=48, node_feature_count=1, node_embedding_dim=9,
            output_dim=105, num_convs=3, conv="GCN", readout=readout,
        )
        total = sum(p.numel() for p in model.parameters())
        return sum(p.numel() for p in model.fc.parameters()) / total

    assert readout_share("mean") < 0.85
    assert readout_share("dim_attention") < 0.95


def test_no_projection_uses_pooled_width():
    """Without the projection the sample embedding is the pooled node vector."""
    projected = flexGCN(node_count=48, node_feature_count=1,
                        node_embedding_dim=16, output_dim=105, conv="GCN",
                        readout="mean", project=True)
    direct = flexGCN(node_count=48, node_feature_count=1,
                     node_embedding_dim=16, output_dim=105, conv="GCN",
                     readout="mean", project=False)
    assert projected.output_dim == 105
    assert direct.output_dim == 16

    x = torch.randn(2, 48, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    assert projected(x, edge_index).shape == (2, 105)
    assert direct(x, edge_index).shape == (2, 16)


def test_no_projection_moves_capacity_into_the_graph():
    """The point: most parameters should end up in the convolutions."""
    def graph_share(**kwargs):
        model = flexGCN(node_count=48, node_feature_count=1,
                        node_embedding_dim=16, output_dim=105, num_convs=3,
                        conv="GCN", **kwargs)
        total = sum(p.numel() for p in model.parameters())
        convs = sum(p.numel() for c in model.convs for p in c.parameters())
        return convs / total

    # The rest is batch norm; the readout itself is parameter-free here.
    assert graph_share(readout="mean", project=False) > 0.8


def test_attention_is_permutation_equivariant():
    """Attention scores nodes by content, so relabelling them changes nothing."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    perm = torch.tensor([4, 3, 2, 1, 0])
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))

    model = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="attention",
                    dropout_rate=0.0).eval()
    assert torch.allclose(model(x, edge_index),
                          model(x[:, perm], inverse[edge_index]), atol=1e-5)


def test_attention_weights_are_a_distribution_over_nodes():
    model = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="attention",
                    dropout_rate=0.0).eval()
    model(torch.randn(2, 5, 1), torch.tensor([[0, 1], [1, 2]]))
    weights = model.last_attention
    assert weights.shape == (2, 5, 1)
    assert torch.allclose(weights.sum(dim=1).squeeze(), torch.ones(2), atol=1e-5)


def test_attention_can_weight_nodes_unequally():
    """Unlike mean pooling, attention is not forced to weight every node 1/N."""
    torch.manual_seed(0)
    model = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="attention",
                    dropout_rate=0.0).eval()
    with torch.no_grad():
        model.attention.weight.mul_(50.0)
    model(torch.randn(2, 5, 1), torch.tensor([[0, 1], [1, 2]]))
    spread = model.last_attention.squeeze(-1).std(dim=1)
    assert (spread > 1e-3).all()


def test_dim_attention_width_is_the_node_count():
    """One number per gene, so the sample vector is as wide as the graph."""
    model = flexGCN(node_count=48, node_feature_count=1, node_embedding_dim=16,
                    output_dim=105, conv="GCN", readout="dim_attention",
                    project=False)
    assert model.output_dim == 48
    x = torch.randn(2, 48, 1)
    assert model(x, torch.tensor([[0, 1], [1, 2]])).shape == (2, 48)


def test_dim_attention_normalises_over_dimensions_not_nodes():
    model = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="dim_attention",
                    dropout_rate=0.0).eval()
    model(torch.randn(2, 5, 1), torch.tensor([[0, 1], [1, 2]]))
    weights = model.last_attention
    assert weights.shape == (2, 5, 4)
    # each node's weights form a distribution across its 4 dimensions
    assert torch.allclose(weights.sum(dim=2), torch.ones(2, 5), atol=1e-5)


def test_dim_attention_keeps_gene_identity():
    """Unlike node pooling, relabelling nodes must change the output."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    perm = torch.tensor([4, 3, 2, 1, 0])
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))

    model = flexGCN(node_count=5, node_feature_count=1, node_embedding_dim=4,
                    output_dim=3, conv="GCN", readout="dim_attention",
                    project=False, dropout_rate=0.0).eval()
    assert not torch.allclose(model(x, edge_index),
                              model(x[:, perm], inverse[edge_index]))


def test_dim_attention_is_one_value_per_gene():
    def width(readout):
        return flexGCN(node_count=490, node_feature_count=1,
                       node_embedding_dim=146, output_dim=105, conv="GCN",
                       readout=readout, project=False).output_dim
    # One value per gene, not node_embedding_dim values per gene.
    assert width("dim_attention") == 490
    assert width("mean") == 146

