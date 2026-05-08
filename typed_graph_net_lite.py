# graphcast/typed_graph_net_lite.py
from typing import Callable, Mapping, Optional, Union

import torch

from graphcast import typed_graph_lite as typed_graph


def segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Equivalent PyTorch de jraph.segment_sum.

    data:
        [num_edges, B, C] ou [num_edges, C]

    segment_ids:
        [num_edges]

    output:
        [num_segments, B, C] ou [num_segments, C]
    """
    segment_ids = segment_ids.long().to(data.device)

    output_shape = (num_segments,) + data.shape[1:]
    output = torch.zeros(
        output_shape,
        dtype=data.dtype,
        device=data.device,
    )

    output.index_add_(0, segment_ids, data)

    return output


def GraphNetwork(
    update_edge_fn: Mapping[str, Callable],
    update_node_fn: Mapping[str, Callable],
    update_global_fn: Optional[Callable] = None,
    aggregate_edges_for_nodes_fn: Callable = segment_sum,
):
    """
    Version Lite PyTorch de GraphNetwork.

    Pipeline :
        1. update edges
        2. update nodes
        3. update globals optionnel
    """

    def _apply_graph_net(
        graph: typed_graph.TypedGraph,
    ) -> typed_graph.TypedGraph:

        updated_graph = graph

        # ======================
        # 1. Edge update
        # ======================
        updated_edges = dict(updated_graph.edges)

        for edge_set_name, edge_fn in update_edge_fn.items():
            edge_set_key = updated_graph.edge_key_by_name(edge_set_name)

            updated_edges[edge_set_key] = _edge_update(
                updated_graph,
                edge_fn,
                edge_set_key,
            )

        updated_graph = updated_graph._replace(edges=updated_edges)

        # ======================
        # 2. Node update
        # ======================
        updated_nodes = dict(updated_graph.nodes)

        for node_set_key, node_fn in update_node_fn.items():
            updated_nodes[node_set_key] = _node_update(
                updated_graph,
                node_fn,
                node_set_key,
                aggregate_edges_for_nodes_fn,
            )

        updated_graph = updated_graph._replace(nodes=updated_nodes)

        # ======================
        # 3. Global update optionnel
        # ======================
        if update_global_fn is not None:
            updated_context = _global_update(
                updated_graph,
                update_global_fn,
            )

            updated_graph = updated_graph._replace(
                context=updated_context
            )

        return updated_graph

    return _apply_graph_net


def _edge_update(
    graph: typed_graph.TypedGraph,
    edge_fn: Callable,
    edge_set_key: typed_graph.EdgeSetKey,
) -> typed_graph.EdgeSet:
    """
    Met à jour les features des arêtes.

    Logique :
        edge_new = MLP(edge_features, sender_node_features, receiver_node_features)
    """

    sender_node_set_name = edge_set_key.node_sets[0]
    receiver_node_set_name = edge_set_key.node_sets[1]

    sender_nodes = graph.nodes[sender_node_set_name]
    receiver_nodes = graph.nodes[receiver_node_set_name]
    edge_set = graph.edges[edge_set_key]

    senders = edge_set.indices.senders.long().to(edge_set.features.device)
    receivers = edge_set.indices.receivers.long().to(edge_set.features.device)

    sender_features = sender_nodes.features[senders]
    receiver_features = receiver_nodes.features[receivers]

    new_edge_features = edge_fn(
        edge_set.features,
        sender_features,
        receiver_features,
    )

    return edge_set._replace(features=new_edge_features)


def _node_update(
    graph: typed_graph.TypedGraph,
    node_fn: Callable,
    node_set_key: str,
    aggregation_fn: Callable,
) -> typed_graph.NodeSet:
    """
    Met à jour les features des nœuds.

    Logique :
        messages reçus = somme des edge_features entrantes
        node_new = MLP(node_features, messages reçus)
    """

    node_set = graph.nodes[node_set_key]
    node_features = node_set.features

    num_nodes = node_features.shape[0]

    sent_features = {}
    received_features = {}

    for edge_set_key, edge_set in graph.edges.items():
        edge_features = edge_set.features

        sender_node_set_name = edge_set_key.node_sets[0]
        receiver_node_set_name = edge_set_key.node_sets[1]

        senders = edge_set.indices.senders.long().to(edge_features.device)
        receivers = edge_set.indices.receivers.long().to(edge_features.device)

        if sender_node_set_name == node_set_key:
            sent_features[edge_set_key.name] = aggregation_fn(
                edge_features,
                senders,
                num_nodes,
            )

        if receiver_node_set_name == node_set_key:
            received_features[edge_set_key.name] = aggregation_fn(
                edge_features,
                receivers,
                num_nodes,
            )

    new_node_features = node_fn(
        node_features,
        sent_features,
        received_features,
    )

    return node_set._replace(features=new_node_features)


def _global_update(
    graph: typed_graph.TypedGraph,
    global_fn: Callable,
) -> typed_graph.Context:
    """
    Global update optionnel.

    Dans GraphCast Lite V1, tu peux ne pas l'utiliser.
    """
    new_global_features = global_fn(
        graph.nodes,
        graph.edges,
        graph.context.features,
    )

    return graph.context._replace(features=new_global_features)


def InteractionNetwork(
    update_edge_fn: Mapping[str, Callable],
    update_node_fn: Mapping[str, Callable],
    aggregate_edges_for_nodes_fn: Callable = segment_sum,
    include_sent_messages_in_node_update: bool = False,
):
    """
    Version Lite de InteractionNetwork.

    C'est le bloc utilisé dans GraphCast :
        Edge update
        Node update

    Si include_sent_messages_in_node_update=False :
        node_fn(node_features, received_features)

    Si True :
        node_fn(node_features, sent_features, received_features)
    """

    wrapped_update_edge_fn = {
        name: fn
        for name, fn in update_edge_fn.items()
    }

    if include_sent_messages_in_node_update:
        wrapped_update_node_fn = {
            name: fn
            for name, fn in update_node_fn.items()
        }
    else:
        wrapped_update_node_fn = {
            name: (
                lambda node_features, sent_features, received_features, fn=fn:
                fn(node_features, received_features)
            )
            for name, fn in update_node_fn.items()
        }

    return GraphNetwork(
        update_edge_fn=wrapped_update_edge_fn,
        update_node_fn=wrapped_update_node_fn,
        update_global_fn=None,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
    )


def GraphMapFeatures(
    embed_edge_fn: Optional[Mapping[str, Callable]] = None,
    embed_node_fn: Optional[Mapping[str, Callable]] = None,
    embed_global_fn: Optional[Callable] = None,
):
    """
    Applique des MLP séparés aux features :
        nodes
        edges
        globals

    Pas de message passing ici.
    """

    def _embed(
        graph: typed_graph.TypedGraph,
    ) -> typed_graph.TypedGraph:

        updated_edges = dict(graph.edges)

        if embed_edge_fn is not None:
            for edge_set_name, embed_fn in embed_edge_fn.items():
                edge_set_key = graph.edge_key_by_name(edge_set_name)
                edge_set = graph.edges[edge_set_key]

                updated_edges[edge_set_key] = edge_set._replace(
                    features=embed_fn(edge_set.features)
                )

        updated_nodes = dict(graph.nodes)

        if embed_node_fn is not None:
            for node_set_key, embed_fn in embed_node_fn.items():
                node_set = graph.nodes[node_set_key]

                updated_nodes[node_set_key] = node_set._replace(
                    features=embed_fn(node_set.features)
                )

        updated_context = graph.context

        if embed_global_fn is not None:
            updated_context = updated_context._replace(
                features=embed_global_fn(updated_context.features)
            )

        return graph._replace(
            edges=updated_edges,
            nodes=updated_nodes,
            context=updated_context,
        )

    return _embed