import pytest

from pinder.eval.dockq import BiotiteDockQ


def test_dockq(dockq_directory):
    native = list(dockq_directory.glob("*.pdb"))[0]
    models = list((dockq_directory / "models").glob("*.pdb"))
    self = BiotiteDockQ(native, models, backbone_definition="dockq", parallel_io=False)
    metrics = self.calculate()
    # Two missing atoms in one of the decoys. DockQ should be the same though.
    assert len(set(metrics.DockQ)) == 1
    assert round(metrics.DockQ.values[0], 2) == 0.39


# ./DockQ.py examples/model.pdb examples/native.pdb
example_expected = {
    "Fnat": 0.533,
    "iRMS": 1.232,
    "LRMS": 1.516,
    "DockQ": 0.700,
}

# ./DockQ.py examples/model2.pdb examples/native2.pdb
example_expected2 = {
    "Fnat": 0.589,
    "iRMS": 1.933,
    "LRMS": 6.985,
    "DockQ": 0.521,
}


@pytest.mark.parametrize(
    "test_system,expected",
    [
        ("dockq_example", example_expected),
        ("dockq_example2", example_expected2),
    ],
)
def test_dockq_example_parity(test_system, expected, test_dir, tolerance=0.014):
    dockq_directory = test_dir / test_system
    native = list(dockq_directory.glob("*.pdb"))[0]
    models = list((dockq_directory / "models").glob("*.pdb"))
    self = BiotiteDockQ(
        native,
        models,
        backbone_definition="dockq",
        parallel_io=False,
    )
    metrics = self.calculate()
    metric_dict = metrics.to_dict(orient="records")[0]
    for measure, value in expected.items():
        delta = abs(value - metric_dict[measure])
        assert delta < tolerance


# ./DockQ.py examples/1A2K_r_l_b.model.pdb examples/1A2K_r_l_b.pdb -native_chain1 A B -model_chain1 A B -native_chain2 C -model_chain2 C -no_needle
multichain_expected = {
    "Fnat": 0.000,
    # Calculation in source code doesn't make sense.
    # AB is the interface, but should be AB-C
    "iRMS": 17.580,
    "LRMS": 56.725,
    "DockQ": 0.010,
}

# ./DockQ.py examples/1A2K_r_l_b.model.pdb examples/1A2K_r_l_b.pdb -native_chain1 A -model_chain1 A -native_chain2 C -model_chain2 C -no_needle
multichain_AC_expected = {
    "Fnat": 0.000,
    "iRMS": 9.676,
    "LRMS": 28.462,
    "DockQ": 0.035,
}

# ./DockQ.py examples/1A2K_r_l_b.model.pdb examples/1A2K_r_l_b.pdb -native_chain1 B -model_chain1 B -native_chain2 C -model_chain2 C -no_needle
multichain_BC_expected = {
    "Fnat": 0.000,
    "iRMS": 15.246,
    "LRMS": 30.770,
    "DockQ": 0.027,
}


@pytest.mark.parametrize(
    "test_system,r_chains,l_chains,expected",
    [
        ("dockq_1A2K", ["A", "B"], ["C"], multichain_expected),
        ("dockq_1A2K", ["C"], ["A"], multichain_AC_expected),
        ("dockq_1A2K", ["C"], ["B"], multichain_BC_expected),
    ],
)
def test_dockq_multichain_parity(test_system, r_chains, l_chains, expected, test_dir):
    dockq_directory = test_dir / test_system
    native = list(dockq_directory.glob("*.pdb"))[0]
    models = list((dockq_directory / "models").glob("*.pdb"))
    self = BiotiteDockQ(
        native,
        models,
        decoy_ligand_chain=l_chains,
        decoy_receptor_chain=r_chains,
        native_ligand_chain=l_chains,
        native_receptor_chain=r_chains,
        backbone_definition="dockq",
        parallel_io=False,
    )
    metrics = self.calculate()
    metric_dict = metrics.to_dict(orient="records")[0]
    for measure, value in expected.items():
        delta = abs(value - metric_dict[measure])
        # TODO: figure out the reason for discrepancy in this case
        if len(r_chains) > 1 and measure == "iRMS":
            tolerance = 1.15
        else:
            tolerance = 0.01
        assert delta < tolerance


def test_dockgpt_regression(pinder_method_test_dir):
    expected = {
        "model_name": "model_0",
        "native_name": "6s0a__A1_P27918--6s0a__B1_P27918",
        "system": "6s0a__A1_P27918--6s0a__B1_P27918",
        "method": "dockgpt",
        "model_folder": "predicted_decoys",
        "iRMS": 7.0187530517578125,
        "LRMS": 20.49225616455078,
        "Fnat": 0.0,
        "DockQ": 0.06349116936326027,
        "CAPRI": "Incorrect",
        "decoy_contacts": 95,
        "native_contacts": 83,
        "initial_decoy_shape": 2704,
        "final_decoy_shape": 2704,
        "initial_native_shape": 2705,
        "final_native_shape": 2705,
    }
    eval_dir = pinder_method_test_dir / "dockgpt"
    system = eval_dir / "6s0a__A1_P27918--6s0a__B1_P27918"
    native = system / f"{system.stem}.pdb"
    decoys = list((system / "predicted_decoys").glob("*.pdb"))
    self = BiotiteDockQ(
        native=native,
        decoys=decoys,
        native_receptor_chain=["R"],
        native_ligand_chain=["L"],
        decoy_receptor_chain=["A"],
        decoy_ligand_chain=["B"],
        parallel_io=False,
    )
    metrics = self.calculate()
    max_dockq = metrics.DockQ.max()
    min_dockq = metrics.DockQ.min()

    assert max_dockq == pytest.approx(0.192343, abs=1e-5)
    assert min_dockq == pytest.approx(0.021724, abs=1e-5)
    metric_dict = metrics.query('model_name == "model_0"').to_dict(orient="records")[0]
    for measure, value in expected.items():
        actual = metric_dict[measure]
        if isinstance(value, str):
            assert actual == value
        else:
            assert actual == pytest.approx(value, abs=1e-5)


def test_diffdock_regression(pinder_method_test_dir):
    expected = {
        "model_name": "model_0",
        "native_name": "5o2z__A1_P06396--5o2z__B1_P06396",
        "system": "5o2z__A1_P06396--5o2z__B1_P06396",
        "method": "diffdock-pp",
        "model_folder": "apo_decoys",
        "iRMS": 36.35594177246094,
        "LRMS": 73.7579574584961,
        "Fnat": 0.0,
        "DockQ": 0.004935333505272865,
        "CAPRI": "Incorrect",
        "decoy_contacts": 0,
        "native_contacts": 158,
        "initial_decoy_shape": 1768,
        "final_decoy_shape": 1768,
        "initial_native_shape": 1670,
        "final_native_shape": 1670,
    }
    eval_dir = pinder_method_test_dir / "diffdock-pp"
    system = eval_dir / "5o2z__A1_P06396--5o2z__B1_P06396"
    native = system / f"{system.stem}.pdb"
    decoys = list((system / "apo_decoys").glob("*.pdb"))
    self = BiotiteDockQ(
        native=native,
        decoys=decoys,
        native_receptor_chain=["R"],
        native_ligand_chain=["L"],
        decoy_receptor_chain=["R"],
        decoy_ligand_chain=["L"],
        parallel_io=False,
    )
    metrics = self.calculate()

    max_dockq = metrics.DockQ.max()
    assert metrics.shape[0] == 1
    metric_dict = metrics.to_dict(orient="records")[0]
    for measure, value in expected.items():
        actual = metric_dict[measure]
        if isinstance(value, str):
            assert actual == value
        else:
            assert actual == pytest.approx(value, abs=1e-5)


def test_geodock_regression(pinder_method_test_dir):
    expected = {
        "model_name": "model_1",
        "native_name": "2e31__A1_Q80UW2--2e31__B1_P63208",
        "system": "2e31__A1_Q80UW2--2e31__B1_P63208",
        "method": "geodock",
        "model_folder": "holo_decoys",
        "iRMS": 7.706840,
        "LRMS": 21.733774,
        "Fnat": 0.03,
        "DockQ": 0.066388,
        "CAPRI": "Incorrect",
    }
    eval_dir = pinder_method_test_dir / "geodock"
    system = eval_dir / "2e31__A1_Q80UW2--2e31__B1_P63208"
    native = system / f"{system.stem}.pdb"
    decoys = list((system / "holo_decoys").glob("*.pdb"))
    self = BiotiteDockQ(
        native=native,
        decoys=decoys,
        native_receptor_chain=["R"],
        native_ligand_chain=["L"],
        decoy_receptor_chain=["R"],
        decoy_ligand_chain=["L"],
        parallel_io=True,
        max_workers=2,
    )
    metrics = self.calculate()
    metric_dict = metrics.to_dict(orient="records")[0]
    for measure, value in expected.items():
        actual = metric_dict[measure]
        if isinstance(value, str):
            assert actual == value
        else:
            assert actual == pytest.approx(value, abs=1e-5)


def test_hdock_irms_not_nan(pinder_method_test_dir):
    expected = {
        "model_name": "6x1g__A1_B3CVM3-R--6x1g__C1_P63000-L.model_30",
        "native_name": "6x1g__A1_B3CVM3--6x1g__C1_P63000",
        "system": "6x1g__A1_B3CVM3--6x1g__C1_P63000",
        "method": "hdock",
        "model_folder": "holo_decoys",
        "iRMS": 8.175938,
        "LRMS": 15.595024,
        "Fnat": 0.023077,
        "DockQ": 0.094892,
        "CAPRI": "Incorrect",
    }
    eval_dir = pinder_method_test_dir / "hdock"
    system = eval_dir / "6x1g__A1_B3CVM3--6x1g__C1_P63000"
    native = system / f"{system.stem}.pdb"
    decoys = list((system / "holo_decoys").glob("*.pdb"))
    self = BiotiteDockQ(
        native=native,
        decoys=decoys,
        native_receptor_chain=["R"],
        native_ligand_chain=["L"],
        decoy_receptor_chain=["R"],
        decoy_ligand_chain=["L"],
        parallel_io=True,
        max_workers=2,
    )
    metrics = self.calculate()
    metric_dict = metrics.to_dict(orient="records")[0]
    for measure, value in expected.items():
        actual = metric_dict[measure]
        if isinstance(value, str):
            assert actual == value
        else:
            assert actual == pytest.approx(value, abs=1e-5)
