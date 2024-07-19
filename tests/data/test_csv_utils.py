import pandas as pd
from pinder.data.csv_utils import read_csv_non_default_na


def test_read_csv_non_default_na(pinder_data_cp, tmp_path):
    pqt_file = pinder_data_cp / "pinder/mappings/6tz5__NA1_P53990-R.parquet"
    df = pd.read_parquet(pqt_file)
    csv_file = tmp_path / f"{pqt_file.stem}.csv.gz"
    df.to_csv(csv_file, index=False)
    mapping = read_csv_non_default_na(csv_file)
    assert set(mapping.asym_id) == {"NA"}
