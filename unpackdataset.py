import zipfile
from pathlib import Path


with zipfile.ZipFile("dataset/dataset.zip","r") as zip_ref:
    zip_ref.extractall("dataset/")

p = Path('dataset/all-samples/')
for f in p.glob('*.zip'):
    with zipfile.ZipFile(f, 'r') as archive:
        archive.extractall(path='dataset/all-samples/')