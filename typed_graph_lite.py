# graphcast/typed_graph_lite.py
from typing import NamedTuple, Tuple, Mapping
import torch


class NodeSet(NamedTuple):
    n_node: torch.Tensor
    features: torch.Tensor


class EdgesIndices(NamedTuple):
    senders: torch.Tensor
    receivers: torch.Tensor


class EdgeSet(NamedTuple):
    n_edge: torch.Tensor
    indices: EdgesIndices
    features: torch.Tensor


class Context(NamedTuple):
    n_graph: torch.Tensor
    features: torch.Tensor


class EdgeSetKey(NamedTuple):
    name: str
    node_sets: Tuple[str, str]


class TypedGraph(NamedTuple):
    context: Context
    nodes: Mapping[str, NodeSet]
    edges: Mapping[EdgeSetKey, EdgeSet]

    def edge_key_by_name(self, name: str) -> EdgeSetKey:
        found_key = [k for k in self.edges.keys() if k.name == name]

        if len(found_key) != 1:
            raise KeyError(
                f"Invalid edge key '{name}'. "
                f"Available edges: {[x.name for x in self.edges.keys()]}"
            )

        return found_key[0]

    def edge_by_name(self, name: str) -> EdgeSet:
        return self.edges[self.edge_key_by_name(name)]