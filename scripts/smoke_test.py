"""CPU integration test using synthetic PNGs and the real ResNet101 (no downloaded weights)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(work):
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from PIL import Image
    torch.set_num_threads(2)
    images = work/'images'
    images.mkdir()
    records = []
    rng = np.random.default_rng(42)
    for i in range(12):
        Image.fromarray(rng.integers(0,256,(32,32),dtype=np.uint8)).save(images/f'{i}.png')
        records.append({'patient_id':str(i),'image_id':str(i),'laterality':'L','cancer':int(i < 4)})
    labels = work/'train.csv'
    pd.DataFrame(records).to_csv(labels,index=False)
    config = {'experiment_name':'synthetic_cpu_smoke','seed':42,'device':'cpu','output_dir':str(work/'run'),
        'data':{'image_dir':str(images),'label_csv_path':str(labels),'split_path':str(work/'split.json'),'image_size':32,'num_workers':0},
        'model':{'name':'resnet101','imagenet_weights':False}, 'training':{'batch_size':2,'freeze_all_batchnorm':True},
        'optimizer':{'type':'adamw','lr':.001,'weight_decay':.0001}, 'loss':{'type':'bce_pos','sqrt_pos_weight':True},
        'scheduler':{'type':'none'}, 'early_stopping':{'monitor':'val_loss','mode':'min'},
        'save_best':{'monitor':'val_auc','mode':'max'}, 'phases':[{'name':'full','loader':'train','epochs':1,'freeze_backbone':True}],
        'logging':{'use_wandb':False}}
    config_path = work/'smoke.yaml'
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLBACKEND':'Agg'}
    command = [sys.executable,str(ROOT/'scripts'/'train.py'),'--config',str(config_path)]
    # Launch outside the project root, including the check path.
    subprocess.run(command+['--check'],cwd=work,env=env,check=True)
    assert not (work/'split.json').exists() and not (work/'run').exists()
    subprocess.run(command,cwd=work,env=env,check=True)
    from src.data.dataset import build_dataloaders
    from src.models.resnet import get_model
    from src.training.experiment_runner import load_initial_checkpoint
    from src.training.evaluate import evaluate
    from src.training.losses import build_criterion
    loaders = build_dataloaders(config)
    model = get_model(imagenet_weights=False)
    result = json.loads((work/'run'/'final_metrics.json').read_text())
    checkpoint = Path(result['selected_checkpoint_path'])
    loaded = load_initial_checkpoint(model,checkpoint,'cpu')
    criterion,_ = build_criterion(config['loss'],'cpu',train_loader=loaders['train'])
    measured = evaluate(model,loaders['val'],criterion,'cpu')
    for key,value in measured.items():
        assert abs(float(result[key])-float(value)) < 1e-6, (key,result[key],value)
    assert loaded['epoch'] == result['best_epoch'] == 1
    assert result['best_phase'] == 'full' and checkpoint.name == 'best_auc.pth'
    # A repeated run must reject stale output and leave the metrics unchanged.
    before = (work/'run'/'final_metrics.json').read_bytes()
    repeated = subprocess.run(command,cwd=work,env=env,capture_output=True,text=True)
    assert repeated.returncode != 0 and 'not empty' in repeated.stderr
    assert (work/'run'/'final_metrics.json').read_bytes() == before
    evidence = {'status':'passed','model':'ResNet101','device':'cpu','epochs':1,'train_n':len(loaders['train'].dataset),'val_n':len(loaders['val'].dataset),'checkpoint_reevaluation':'all metrics match within 1e-6','outside_project_cwd':True,'preflight_no_files_written':True,'stale_output_rejected':True,'weights':'random initialization; synthetic data; no research performance claim', 'python':sys.version.split()[0],'torch':str(torch.__version__)}
    (work/'smoke_result.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,help='New directory for test evidence; omitted uses a cleaned temporary directory.')
    args = parser.parse_args()
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True,exist_ok=False)
        run(work)
    else:
        with tempfile.TemporaryDirectory(prefix='mammography_smoke_') as tmp:
            run(Path(tmp))


if __name__ == '__main__': main()
