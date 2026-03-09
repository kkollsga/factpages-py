import os

def test_export_to_json(fp, tmp_path):
    fp.db.export_to_json(tmp_path)
    files = os.listdir(tmp_path)
    assert "sub_area.json" in files
    # TODO: check if all the other expected files are generated
    # TODO: check if the content of the files is as expected.

def test_export_to_csv(fp, tmp_path):
    fp.db.export_to_csv(tmp_path)
    files = os.listdir(tmp_path)
    assert "sub_area.csv" in files
    # TODO: as in json

def test_export_to_parquet(fp, tmp_path):
    fp.db.export_to_parquet(tmp_path)
    files = os.listdir(tmp_path)
    assert "sub_area.parquet" in files
    # TODO: as in json


