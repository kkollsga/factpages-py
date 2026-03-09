import os

def test_export_to_json(fp, tmp_path):
    fp.db.export_to_json(tmp_path)
    files = sorted(os.listdir(tmp_path))
    assert files == ['company.json', 'discovery.json', 'facility.json', 'field.json', 'licence.json', 'wellbore.json']
    # TODO: check if all the other expected files are generated
    # TODO: check if the content of the files is as expected.

def test_export_to_csv(fp, tmp_path):
    fp.db.export_to_csv(tmp_path)
    files = sorted(os.listdir(tmp_path))
    assert files == ['company.csv', 'discovery.csv', 'facility.csv', 'field.csv', 'licence.csv', 'wellbore.csv']
    # TODO: as in json

def test_export_to_parquet(fp, tmp_path):
    fp.db.export_to_parquet(tmp_path)
    files = sorted(os.listdir(tmp_path))
    assert files == ['company.parquet', 'discovery.parquet', 'facility.parquet', 'field.parquet', 'licence.parquet', 'wellbore.parquet']
    # TODO: as in json


