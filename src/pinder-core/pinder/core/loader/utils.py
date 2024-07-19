import numpy as np
import networkx as nx
from numpy.typing import NDArray
from networkx.classes.graph import Graph


def create_nx_radius_graph(
    coordinates: NDArray[np.double], radius: float = 15.0
) -> Graph:
    pairwise_distances = np.sqrt(((coordinates[:, None, :] - coordinates) ** 2).sum(-1))
    shape = pairwise_distances.shape
    pairwise_distances = pairwise_distances.flatten()
    pairwise_distances[pairwise_distances < radius] = 1
    pairwise_distances[pairwise_distances >= radius] = 0
    pairwise_distances = pairwise_distances.reshape(shape)
    np.fill_diagonal(pairwise_distances, 0)
    return nx.from_numpy_array(pairwise_distances)
