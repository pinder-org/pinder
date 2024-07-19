import pytest
from collections import Counter

from pinder.data.pipeline.scatter import (
    chunk_all_vs_all_indices,
    chunk_collection,
    chunk_dict,
    chunk_dict_with_indices,
)


@pytest.mark.parametrize(
    "nitems, batch_size, num_batches, expected_chunks, expected_chunk_size",
    [
        (158, 200, 50, 1, 158),
        (1010, 200, 50, 6, 169),
        (1010, 100, 50, 11, 92),
        (10000, 200, 50, 50, 200),
        (10000, 100, 100, 100, 100),
    ],
)
def test_chunk_collection(
    nitems,
    batch_size,
    num_batches,
    expected_chunks,
    expected_chunk_size,
):
    items = [str(i) for i in range(nitems)]
    chunks = chunk_collection(
        items,
        batch_size=batch_size,
        num_batches=num_batches,
    )
    chunk_list = list(chunks)
    assert expected_chunks == len(chunk_list)
    sizes = Counter((len(chunk) for chunk in chunk_list))
    assert len(sizes) <= 2
    assert expected_chunk_size in sizes


@pytest.mark.parametrize(
    "nitems, batch_size, expected_chunks, expected_chunk_size",
    [
        (158, 200, 1, 158),
        (1010, 200, 6, 10),
        (1010, 100, 11, 10),
        (10000, 200, 50, 200),
        (10000, 100, 100, 100),
    ],
)
def test_chunk_dict(
    nitems,
    batch_size,
    expected_chunks,
    expected_chunk_size,
):
    data = {str(i): i for i in range(nitems)}
    chunks = chunk_dict(
        data,
        batch_size=batch_size,
    )
    chunk_list = list(chunks)
    assert expected_chunks == len(chunk_list)
    sizes = Counter((len(chunk) for chunk in chunk_list))
    assert len(sizes) <= 2
    assert expected_chunk_size in sizes


@pytest.mark.parametrize(
    "nitems, batch_size, expected_chunks",
    [
        (158, 200, 1),
        (1010, 200, 6),
        (1010, 100, 11),
        (10000, 200, 50),
        (10000, 100, 100),
    ],
)
def test_chunk_dict_with_indices(
    nitems,
    batch_size,
    expected_chunks,
):
    data = {str(i): i for i in range(nitems)}
    chunks = chunk_dict_with_indices(
        data,
        batch_size=batch_size,
    )
    n_chunks = 0
    for chunk in chunks:
        assert isinstance(chunk, tuple)
        assert len(chunk) == 2
        idx, dict_chunk = chunk
        assert isinstance(idx, int)
        assert len(dict_chunk) <= batch_size
        n_chunks += 1
    assert expected_chunks == n_chunks


@pytest.mark.parametrize(
    "nitems, batch_size, expected_chunks",
    [
        (158, 200, 1),
        (1010, 200, 21),
        (1010, 100, 66),
        (10000, 200, 1275),
        (10000, 100, 5050),
    ],
)
def test_chunk_all_vs_all_indices(
    nitems,
    batch_size,
    expected_chunks,
):
    data = [i for i in range(nitems)]
    chunks = chunk_all_vs_all_indices(data, batch_size)
    n_chunks = 0
    for chunk in chunks:
        assert isinstance(chunk, tuple)
        assert all(isinstance(idx, int) for idx in chunk)
        n_chunks += 1
    assert n_chunks == expected_chunks
