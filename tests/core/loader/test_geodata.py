import pytest

from pinder.core import get_index, PinderSystem
from pinder.core.loader.geodata import PairedPDB, NodeRepresentation
from pinder.core.loader.dataset import get_geo_loader, PPIDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


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

    assert isinstance(loader, DataLoader)
    assert hasattr(loader, "dataset")
    ds = loader.dataset
    assert len(ds) == n_expected_ids
