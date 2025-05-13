import networkx as nx

from gossiplearning.config import Config, NodeConfig
from gossiplearning.models import Link, TimeFn


def add_node_and_links_to_config(
    conf: Config,
    network: nx.Graph,
    round_trip_fn: TimeFn,
    model_transmission_fn: TimeFn,
) -> Config:
    copied_config = Config.model_validate(conf.model_dump())

    copied_config.nodes = tuple(
        NodeConfig(
            id=i,
            links=tuple(
                Link(
                    node=j,
                    weights_transmission_time=model_transmission_fn(i, j),
                    round_trip_time=round_trip_fn(i, j),
                )
                for j, _ in network.adj[i].items()
            ),
        )
        for i in range(conf.n_nodes)
    )

    return copied_config
