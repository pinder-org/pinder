import os

# MacOS workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # noqa
import pytest

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as TorchGeoDataLoader

from pinder.core import get_index, PinderSystem
from pinder.core.loader.geodata import PairedPDB, NodeRepresentation
from pinder.core.loader.dataset import (
    get_geo_loader,
    get_torch_loader,
    PinderDataset,
    PPIDataset,
)

EXPECTED_STRUCTURE_KEYS = {
    "atom_types",
    "residue_types",
    "atom_coordinates",
    "residue_coordinates",
    "residue_ids",
}


def test_pairedpdb_heterodata(pinder_temp_dir):
    pinder_id = "3s9d__B1_P48551--3s9d__A1_P01563"
    system = PinderSystem(pinder_id)

    nodes = {NodeRepresentation("atom"), NodeRepresentation("residue")}
    holo_data = PairedPDB.from_pinder_system(
        system=system,
        monomer1="holo_receptor",
        monomer2="holo_ligand",
        node_types=nodes,
    )
    assert isinstance(holo_data, HeteroData)

    expected_node_types = [
        "ligand_residue",
        "receptor_residue",
        "ligand_atom",
        "receptor_atom",
    ]
    assert holo_data.num_nodes == 2780
    assert holo_data.num_edges == 0
    assert isinstance(holo_data.num_node_features, dict)
    expected_num_feats = {
        "ligand_residue": 0,
        "receptor_residue": 0,
        "ligand_atom": 12,
        "receptor_atom": 12,
    }
    for k, v in expected_num_feats.items():
        assert holo_data.num_node_features[k] == v

    assert holo_data.node_types == expected_node_types

    apo_data = PairedPDB.from_pinder_system(
        system=system,
        monomer1="apo_receptor",
        monomer2="apo_ligand",
        node_types=nodes,
    )
    assert isinstance(apo_data, HeteroData)

    assert apo_data.num_nodes == 3437
    assert apo_data.num_edges == 0
    assert isinstance(apo_data.num_node_features, dict)
    expected_num_feats = {
        "ligand_residue": 0,
        "receptor_residue": 0,
        "ligand_atom": 12,
        "receptor_atom": 12,
    }
    for k, v in expected_num_feats.items():
        assert apo_data.num_node_features[k] == v

    assert apo_data.node_types == expected_node_types


@pytest.mark.parametrize(
    "split, ids, parallel, limit_by",
    [
        ("train", None, False, 5),
        (None, ["7mbm__I1_P84233--7mbm__J1_P62799"], True, None),
    ],
)
def test_ppi_dataset(split, ids, parallel, limit_by, pinder_temp_dir, capfd):
    nodes = {NodeRepresentation("atom"), NodeRepresentation("residue")}
    train_dataset = PPIDataset(
        node_types=nodes,
        split=split,
        ids=ids,
        monomer1="holo_receptor",
        monomer2="holo_ligand",
        limit_by=limit_by,
        force_reload=True,
        max_workers=1,
        parallel=parallel,
    )
    pindex = get_index()
    raw_ids = set(train_dataset.raw_file_names)
    if limit_by:
        n_expected_ids = limit_by
    elif ids:
        n_expected_ids = len(ids)
    else:
        n_expected_ids = len(set(pindex.query(f'split == "{split}"').id))
    assert len(train_dataset) == n_expected_ids
    assert len(raw_ids.intersection(set(pindex.id))) == n_expected_ids
    processed_ids = {f.stem for f in train_dataset.processed_file_names}
    assert len(processed_ids.intersection(set(pindex.id))) == n_expected_ids

    data_item = train_dataset[0]
    assert isinstance(data_item, HeteroData)

    data_item = train_dataset.get_filename("7mbm__I1_P84233--7mbm__J1_P62799")
    assert data_item.num_nodes == 1600
    assert data_item.num_edges == 0
    assert isinstance(data_item.num_node_features, dict)
    expected_num_feats = {
        "ligand_residue": 0,
        "receptor_residue": 0,
        "ligand_atom": 12,
        "receptor_atom": 12,
        "pdb": 0,
    }
    for k, v in expected_num_feats.items():
        assert data_item.num_node_features[k] == v

    expected_node_types = [
        "ligand_residue",
        "receptor_residue",
        "ligand_atom",
        "receptor_atom",
        "pdb",
    ]
    assert data_item.node_types == expected_node_types

    train_dataset.print_summary()
    out, err = capfd.readouterr()
    assert f"PPIDataset (#graphs={n_expected_ids}):" in out

    loader = get_geo_loader(train_dataset)

    assert isinstance(loader, TorchGeoDataLoader)
    assert hasattr(loader, "dataset")
    ds = loader.dataset
    assert len(ds) == n_expected_ids


@pytest.mark.parametrize(
    "split, ids, monomer_priority, expected_len",
    [
        ("train", None, "holo", 1560682),
        ("val", None, "holo", 1958),
        ("test", None, "holo", 1955),
        ("train", ["7mbm__I1_P84233--7mbm__J1_P62799"], "holo", 1),
        (None, ["7mbm__I1_P84233--7mbm__J1_P62799"], "holo", 1),
    ],
)
def test_pinder_dataset(split, ids, monomer_priority, expected_len, pinder_temp_dir):
    dataset = PinderDataset(
        split=split,
        ids=ids,
        monomer_priority=monomer_priority,
    )
    assert len(dataset) == expected_len
    assert len(dataset) == len(dataset.loader)
    data_item = dataset[0]
    assert isinstance(data_item, dict)
    assert isinstance(data_item["target_complex"], dict)
    assert isinstance(data_item["feature_complex"], dict)

    assert set(data_item["target_complex"].keys()) == EXPECTED_STRUCTURE_KEYS
    assert set(data_item["feature_complex"].keys()) == EXPECTED_STRUCTURE_KEYS
    assert isinstance(data_item["target_complex"]["atom_coordinates"], torch.Tensor)
    # Ensure feature and target complex are cropped to same shapes
    assert (
        data_item["feature_complex"]["atom_coordinates"].shape
        == data_item["target_complex"]["atom_coordinates"].shape
    )
    assert data_item["feature_complex"]["atom_coordinates"].shape[1] == 3


@pytest.mark.parametrize(
    "split, monomer_priority, batch_size",
    [
        ("train", "holo", 2),
        ("train", "apo", 1),
        ("train", "pred", 1),
        ("train", "random_mixed", 1),
        ("val", "holo", 1),
        ("test", "holo", 1),
    ],
)
def test_torch_loader(split, monomer_priority, batch_size, pinder_temp_dir):
    dataset = PinderDataset(
        split=split,
        monomer_priority=monomer_priority,
    )
    loader = get_torch_loader(dataset, num_workers=0, batch_size=batch_size)
    assert isinstance(loader, DataLoader)
    assert hasattr(loader, "dataset")
    assert len(loader.dataset) > 0
    batch = next(iter(loader))
    # expected batch dict keys
    assert set(batch.keys()) == {
        "target_complex",
        "feature_complex",
        "id",
        "sample_id",
        "target_id",
    }
    assert isinstance(batch["target_complex"], dict)
    assert isinstance(batch["target_complex"]["atom_coordinates"], torch.Tensor)
    feature_coords = batch["feature_complex"]["atom_coordinates"]
    # Ensure batch size propagates to tensor dims
    assert feature_coords.shape[0] == batch_size
    # Ensure coordinates have dim 3
    assert feature_coords.shape[2] == 3
