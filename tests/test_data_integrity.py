import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image
from src.data.splits import load_rsna_dataframe, get_or_create_group_split, get_or_create_small_subset
from src.data.dataset import build_dataloaders


def groups():
    index = {f'p{i}_L': [i] for i in range(10)}
    labels = {key: int(i < 4) for i, key in enumerate(index)}
    return index, labels


@pytest.mark.parametrize('mutation', ['overlap', 'duplicate', 'stale', 'seed', 'fraction'])
def test_invalid_split_is_rejected_without_overwriting(tmp_path, mutation):
    index, labels = groups()
    path = tmp_path/'split.json'
    get_or_create_group_split(index, labels, path, .2, 42)
    cached = json.loads(path.read_text())
    if mutation == 'overlap':
        cached['val_group_keys'].append(cached['train_group_keys'][0])
    elif mutation == 'duplicate':
        cached['train_group_keys'].append(cached['train_group_keys'][0])
    elif mutation == 'stale':
        cached['train_group_keys'][0] = 'unknown'
    elif mutation == 'seed':
        cached['seed'] = 43
    else:
        cached['val_pct'] = .3
    path.write_text(json.dumps(cached))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        get_or_create_group_split(index, labels, path, .2, 42)
    assert path.read_bytes() == before


def test_subset_reproducible_across_hash_seeds(tmp_path):
    code = '''import json
from pathlib import Path
from src.data.splits import get_or_create_small_subset
keys = {f'p{i}_L' for i in range(80)}
index = {key: [int(key.split('_')[0][1:])] for key in keys}
labels = {key: int(int(key.split('_')[0][1:]) < 15) for key in keys}
_, info = get_or_create_small_subset(keys, set(), index, labels, Path('unused.json'), 3, 42, .5, create_cache=False)
print(json.dumps(info['small_group_keys']))'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': str(seed)}, cwd=Path(__file__).resolve().parents[1], text=True) for seed in (1, 999)]
    assert outputs[0] == outputs[1]


def test_subset_cache_mismatch_does_not_regenerate(tmp_path):
    index, labels = groups()
    path = tmp_path/'small.json'
    get_or_create_small_subset(set(index), set(), index, labels, path, 1, 42)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        get_or_create_small_subset(set(index), set(), index, labels, path, 1, 43)
    assert path.read_bytes() == before


@pytest.mark.parametrize('bad', ['duplicate', 'nonbinary', 'null', 'laterality'])
def test_bad_labels_fail(tmp_path, bad):
    records = [{'patient_id': '01', 'image_id': '9007199254740993', 'laterality': 'L', 'cancer': 0}, {'patient_id': '02', 'image_id': '2', 'laterality': 'R', 'cancer': 1}]
    if bad == 'duplicate': records[1]['image_id'] = records[0]['image_id']
    if bad == 'nonbinary': records[0]['cancer'] = .5
    if bad == 'null': records[0]['patient_id'] = None
    if bad == 'laterality': records[0]['laterality'] = 'unknown'
    path = tmp_path/'train.csv'
    pd.DataFrame(records).to_csv(path, index=False)
    with pytest.raises(ValueError): load_rsna_dataframe(path)


def test_ids_preserve_leading_zero_and_precision(tmp_path):
    path = tmp_path/'train.csv'
    path.write_text('patient_id,image_id,laterality,cancer\n001,9007199254740993,L,1\n')
    df = load_rsna_dataframe(path)
    assert df.patient_id[0] == '001' and df.image_id[0] == '9007199254740993'


def make_data(tmp_path):
    image_dir = tmp_path/'images'
    image_dir.mkdir()
    records = []
    for i in range(12):
        Image.new('L', (32, 32), i*20).save(image_dir/f'{i}.png')
        records.append({'patient_id': str(i), 'image_id': str(i), 'laterality': 'L', 'cancer': int(i < 4)})
    csv = tmp_path/'train.csv'
    pd.DataFrame(records).to_csv(csv, index=False)
    return {'experiment_name': 'test', 'data': {'image_dir': str(image_dir), 'label_csv_path': str(csv), 'split_path': str(tmp_path/'split.json'), 'num_workers': 0, 'image_size': 32}, 'training': {'batch_size': 2}, 'phases': [{'name': 'full', 'epochs': 1}], 'output_dir': str(tmp_path/'output'), 'device': 'cpu', 'model': {'imagenet_weights': False}}


@pytest.mark.parametrize('bad', ['missing', 'extra', 'duplicate'])
def test_png_csv_mismatch_fails(tmp_path, bad):
    cfg = make_data(tmp_path)
    image_dir = Path(cfg['data']['image_dir'])
    if bad == 'missing': (image_dir/'0.png').unlink()
    elif bad == 'extra': Image.new('L', (32,32)).save(image_dir/'extra.png')
    else:
        (image_dir/'sub').mkdir()
        Image.new('L', (32,32)).save(image_dir/'sub'/'0.png')
    with pytest.raises(ValueError): build_dataloaders(cfg)
    assert not Path(cfg['data']['split_path']).exists()


def test_dry_preflight_does_not_create_caches(tmp_path):
    cfg = make_data(tmp_path)
    loaders = build_dataloaders(cfg, create_splits=False)
    assert len(loaders['val'].dataset) > 0
    assert not Path(cfg['data']['split_path']).exists()
