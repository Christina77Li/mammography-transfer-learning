import json
from pathlib import Path
import torch
from torch import nn
import pytest
import yaml
from src.training import experiment_runner as runner
from src.training import checkpoint_policy, checkpointing, metrics as metric_helpers, optimizers, trainer
from src.training.stage_trainer import run_stagewise_rollback_phase
from src.training.stage_selector import StageSelector
from src.training.checkpointing import CheckpointManager, save_checkpoint
from src.training.train_epoch import train_one_epoch
from src.training.evaluate import evaluate
from src.logging.history_logger import HistoryLogger
from src.logging.wandb_logger import WandbLogger
from src.utils.config import validate_config, bn_policy, preflight
from test_data_integrity import make_data


class TinyBackbone(nn.Module):
    """Cheap network with ResNet block names for orchestration regressions."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 1)
        self.bn1 = nn.BatchNorm2d(2)
        for i in range(1, 5): setattr(self, f'layer{i}', nn.Sequential(nn.Conv2d(2,2,1), nn.ReLU()))
        self.fc = nn.Linear(2,1)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        for i in range(1,5): x = getattr(self, f'layer{i}')(x)
        return self.fc(x.mean(dim=(2,3)))


def metrics(loss=.4, auc=.6):
    return {'loss': loss, 'auc': auc, 'pr_auc': .1, 'f1@0.5': .2, 'acc@0.5': .7, 'recall@0.5': .4, 'sensitivity@0.5': .4, 'specificity@0.5': .8, 'best_f1': .3, 'best_th': .3, 'num_pred_pos@0.5': 1, 'prob_pos_mean': .4, 'prob_neg_mean': .2}


def test_reporting_excludes_warmup_and_uses_save_best(tmp_path):
    model = TinyBackbone()
    phases = [{'name':'warmup','loader':'small'}, {'name':'full','loader':'train'}]
    for phase, loss, auc in [('warmup', .1, .99), ('full', .4, .6)]:
        model._checkpoint_context = {'phase': phase, 'stage': 'head'}
        manager = CheckpointManager(tmp_path/'phases'/phase)
        manager.update(model, None, 1, metrics(loss, auc))
    cfg = {'save_best': {'monitor': 'val_auc'}, 'early_stopping': {'monitor':'val_loss'}}
    phase, monitor, path, result = checkpoint_policy.reporting_checkpoint(tmp_path,cfg,phases)
    assert (phase, monitor, result['auc'], result['loss']) == ('full', 'val_auc', .6, .4)
    assert path.name == 'best_auc.pth' and result['context']['phase'] == 'full'


def test_strict_load_and_actual_provenance(tmp_path):
    model = TinyBackbone()
    path = tmp_path/'arbitrary_name.pth'
    model._checkpoint_context = {'phase':'full', 'stage':'layer3'}
    save_checkpoint(path, model, None, 17, metrics())
    actual = checkpointing.load_initial_checkpoint(TinyBackbone(), path, 'cpu')
    assert actual['epoch'] == 17 and actual['context']['stage'] == 'layer3'
    raw = tmp_path/'raw.pth'
    torch.save(model.state_dict(), raw)
    checkpointing.load_initial_checkpoint(TinyBackbone(), raw, 'cpu')
    torch.save({'fc.weight': model.fc.weight}, raw)
    with pytest.raises(RuntimeError): checkpointing.load_initial_checkpoint(TinyBackbone(), raw, 'cpu')


def test_no_eligible_checkpoint_fails_instead_of_violating_guard():
    with pytest.raises(ValueError, match='loss guard'):
        checkpoint_policy.select_guarded([{'val_loss': .4, 'val_auc': .8}], .4, .05, .3)


def test_rollback_checkpoint_matches_selected_weights_metrics_optimizer(tmp_path, monkeypatch):
    cfg = make_data(tmp_path)
    cfg.update({'optimizer': {'type':'adamw', 'lr': .001}, 'stage_transition': {'monitor':'val_loss','mode':'min','min_delta': .001,'patience':10}, 'stagewise_stages': [{'name':'conv1', 'trainable_blocks':['conv1'], 'conv1_lr':.001, 'head_lr':.001, 'min_epochs':1,'max_epochs':3}]})
    loaders = runner.build_dataloaders(cfg)
    model = TinyBackbone()
    evaluation_losses = iter([.4000, .3995, .5])
    original_evaluate = trainer.evaluate
    def controlled_evaluate(*args, **kwargs):
        m = original_evaluate(*args, **kwargs)
        m['loss'] = next(evaluation_losses)
        return m
    monkeypatch.setattr(trainer, 'evaluate', controlled_evaluate)
    output = tmp_path/'run'
    history = HistoryLogger(output/'history.csv')
    epoch, summary = run_stagewise_rollback_phase(cfg, {'name':'full', 'stagewise_rollback': True}, model, loaders['train'], loaders['val'], output, history, WandbLogger(cfg,'test'), 'cpu', 0)
    selected = summary['selected_transition_checkpoints']['conv1']
    assert selected['epoch'] == 2 and selected['stage_best_val_loss'] == .3995
    second_row = next(row for row in history.rows if row.get('stage_epoch') == 2)
    assert second_row['stage_transition_counter'] == 1
    final = torch.load(output/'checkpoints'/'final_conv1_completed.pth', weights_only=True)
    chosen = torch.load(selected['path'], weights_only=True)
    assert epoch == 3 and final['epoch'] == 2 and final['metrics'] == chosen['metrics'] == summary['last_metrics']
    for key, value in final['model_state_dict'].items(): assert torch.equal(value, chosen['model_state_dict'][key])
    for key, state in final['optimizer_state_dict']['state'].items():
        for name, value in state.items(): assert torch.equal(value, chosen['optimizer_state_dict']['state'][key][name])
    restored = TinyBackbone()
    restored.load_state_dict(final['model_state_dict'])
    actual = original_evaluate(model,loaders['val'],nn.BCEWithLogitsLoss(),'cpu')
    recovered = original_evaluate(restored,loaders['val'],nn.BCEWithLogitsLoss(),'cpu')
    assert actual == recovered
    assert summary['subset_size'] == len(loaders['train'].dataset) and summary['val_prevalence'] > 0


@pytest.mark.parametrize('train_affine', [True, False])
def test_batchnorm_stats_and_affine_are_independent(tmp_path, train_affine):
    cfg = make_data(tmp_path)
    cfg['batch_norm'] = {'freeze_running_stats':True,'train_bn_affine':train_affine}
    model = TinyBackbone()
    optimizers.configure_stage_trainability(model,['conv1'],freeze_all_bn=True)
    optimizers.apply_bn_affine_policy(model,cfg)
    assert model.bn1.weight.requires_grad is train_affine
    before = model.bn1.running_mean.clone()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    loaders = runner.build_dataloaders(cfg)
    train_one_epoch(model,loaders['train'],nn.BCEWithLogitsLoss(),opt,'cpu',freeze_all_bn=True)
    assert torch.equal(before,model.bn1.running_mean)
    assert model.bn1.weight.requires_grad is train_affine


@pytest.mark.parametrize('change', ['optimizer', 'device', 'reset', 'rate', 'report'])
def test_invalid_config_fails(tmp_path,change):
    cfg = make_data(tmp_path)
    if change == 'optimizer': cfg['optimizer'] = {'type':'typo'}
    elif change == 'device': cfg['device'] = 'magic'
    elif change == 'report': cfg['save_best'] = {'monitor':'invalid'}
    else:
        cfg['phases'][0]['stagewise_rollback'] = True
        cfg['stagewise_stages'] = [{'name':'layer4','trainable_blocks':['layer4'],'head_lr':.001, 'max_epochs':1}]
        if change == 'reset':
            cfg['stagewise_stages'][0]['layer4_lr'] = .001
            cfg['stage_transition'] = {'reset_optimizer_on_new_stage':False}
    with pytest.raises(ValueError): validate_config(cfg)


def test_experiment_configs_validate():
    root = Path(__file__).resolve().parents[1]
    for path in (root/'configs').glob('*.yaml'):
        validate_config(yaml.safe_load(path.read_text()))


def test_existing_output_is_preserved(tmp_path):
    cfg = make_data(tmp_path)
    output = Path(cfg['output_dir'])
    output.mkdir()
    (output/'keep').write_text('existing')
    with pytest.raises(ValueError,match='not empty'): preflight(cfg)
    assert (output/'keep').read_text() == 'existing'


def test_cpu_override_and_nonfinite_validation():
    assert runner.check_runtime({'device':'cpu'}).type == 'cpu'
    with pytest.raises(ValueError): metric_helpers.checked_metrics(metrics(float('nan')))


@pytest.mark.parametrize('config_name', [
    'exp1_baseline.yaml',
    'exp2a_balanced_warmup.yaml',
    'exp2b_full_data_transfer.yaml',
    'exp3_gradual_unfreezing.yaml',
    'exp4_combined_strategy.yaml',
])
def test_all_experiment_orchestrations_with_small_network(tmp_path,monkeypatch,config_name):
    """Real train/evaluate/rollback loop on each config, shortening epochs and loosening synthetic-only guards."""
    data_cfg = make_data(tmp_path)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs'/config_name).read_text())
    cfg['data'] = data_cfg['data']
    cfg['data']['small_subset'] = {'enabled':True,'pos_neg_ratio':3,'split_path':str(tmp_path/'subset.json')}
    cfg['training']['batch_size'] = 2
    cfg['output_dir'] = str(tmp_path/'run')
    cfg['device'] = 'cpu'
    cfg['model']['imagenet_weights'] = False
    cfg['model']['init_checkpoint'] = None
    cfg['scheduler'] = {'type':'none'}
    for phase in cfg['phases']:
        phase['epochs'] = 1
        if 'scheduler' in phase: phase['scheduler'] = {'type':'none'}
        selector = phase.get('checkpoint_selection')
        if selector:
            selector['max_allowed_val_loss'] = 1000
        transitions = phase.get('stage_transition',cfg.get('stage_transition',{}))
        transitions.pop('max_allowed_val_loss',None)
        for stage in phase.get('stagewise_stages',cfg.get('stagewise_stages',[])):
            stage['min_epochs'] = stage['max_epochs'] = 1
    if 'exp2b' in config_name:
        path = tmp_path/'loaded.pth'
        save_checkpoint(path,TinyBackbone(),None,19,metrics(.7,.55))
        cfg['model']['init_checkpoint'] = str(path)
    config_path = tmp_path/'test.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner,'get_model',lambda **kwargs:TinyBackbone())
    monkeypatch.chdir(tmp_path)
    result = runner.run_experiment(config_path)
    checkpoint = torch.load(result['selected_checkpoint_path'],weights_only=True)
    for key,value in checkpoint['metrics'].items(): assert value == result[key]
    assert result['best_epoch'] == checkpoint['epoch']
    assert result['best_stage'] == checkpoint['context']['stage']
    assert result['best_phase'] == checkpoint['context']['phase']
    if 'exp4' in config_name: assert result['best_phase'] == 'phase3_full_gradual_unfreeze'
    if 'exp2b' in config_name:
        assert result['loaded_checkpoint']['epoch'] == 19
        import pandas as pd
        history = pd.read_csv(tmp_path/'run'/'history.csv')
        assert history['source_epoch'].dropna().eq(19).all()
        init = json.loads((tmp_path/'run'/'initialization.json').read_text())
        assert next(iter(init['historical_reference'].values()))['source_epoch'] == 4


def test_rollback_final_metrics_match_real_reevaluation(tmp_path,monkeypatch):
    cfg = make_data(tmp_path)
    cfg['training']['freeze_all_batchnorm'] = True
    cfg.update({'optimizer':{'type':'adamw','lr':.001},'stage_transition':{'monitor':'val_loss','mode':'min','patience':10},'stagewise_stages':[{'name':'conv1','trainable_blocks':['conv1'],'conv1_lr':.001,'head_lr':.001,'max_epochs':3}]})
    loaders = runner.build_dataloaders(cfg)
    model = TinyBackbone()
    biases = iter([0.,-1.,3.])
    def train_controlled(model,loader,criterion,optimizer,*args,**kwargs):
        optimizer.zero_grad()
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.fill_(next(biases))
        return {'loss':.5,'acc':.5}
    monkeypatch.setattr(trainer,'train_one_epoch',train_controlled)
    output = tmp_path/'run'
    _,summary = run_stagewise_rollback_phase(cfg,{'name':'full'},model,loaders['train'],loaders['val'],output,HistoryLogger(output/'history.csv'),WandbLogger(cfg,'test'),'cpu',0)
    final = torch.load(output/'checkpoints'/'final_conv1_completed.pth',weights_only=True)
    assert final['epoch'] == 2
    measured = evaluate(model,loaders['val'],nn.BCEWithLogitsLoss(),'cpu')
    assert measured == final['metrics'] == summary['last_metrics']
    sidecar = json.loads((output/'checkpoints'/'final_conv1_completed.pth.metrics.json').read_text())
    for key,value in measured.items(): assert sidecar[key] == value


def test_f1_monitor_alias_uses_measured_metric():
    assert metric_helpers.monitor_value(metrics(), 'val_f1') == .2
    assert metric_helpers.monitor_value(metrics(), 'val_f1@0.5') == .2


def test_exp2a_transition_file_is_consumed_by_exp2b(tmp_path, monkeypatch):
    """Run the transfer chain with real loops on a tiny network and synthetic data."""
    import copy
    data_config = make_data(tmp_path)
    root = Path(__file__).resolve().parents[1]
    warmup = yaml.safe_load((root / 'configs/exp2a_balanced_warmup.yaml').read_text())
    transfer = yaml.safe_load((root / 'configs/exp2b_full_data_transfer.yaml').read_text())
    expected_relative = Path(warmup['output_dir']) / 'checkpoints/selected_transition_checkpoint.pth'
    assert Path(transfer['model']['init_checkpoint']) == expected_relative
    expected_path = tmp_path / expected_relative
    monkeypatch.setattr(runner, 'get_model', lambda **kwargs: TinyBackbone())

    def prepare(config, filename):
        subset = config['data'].get('small_subset')
        config['data'] = copy.deepcopy(data_config['data'])
        if subset:
            config['data']['small_subset'] = copy.deepcopy(subset)
            config['data']['small_subset']['split_path'] = str(tmp_path / 'subset.json')
        config['training']['batch_size'] = 2
        config['output_dir'] = str(tmp_path / config['output_dir'])
        config['device'] = 'cpu'
        config['model']['imagenet_weights'] = False
        config['scheduler'] = {'type': 'none'}
        for phase in config['phases']:
            phase['epochs'] = 1
            # Synthetic fixtures exercise file generation/loading, not research loss bounds.
            phase['checkpoint_selection']['max_allowed_val_loss'] = 1000
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(config))
        return path

    warmup_result = runner.run_experiment(prepare(warmup, 'warmup.yaml'))
    selected = warmup_result['phase_summaries'][0]['selected_transition_checkpoint']
    assert Path(selected['path']) == expected_path and expected_path.is_file()
    assert expected_path.with_suffix('.pth.metrics.json').is_file()
    selected_payload = torch.load(expected_path, weights_only=True)

    transfer['model']['init_checkpoint'] = str(expected_path)
    transfer_result = runner.run_experiment(prepare(transfer, 'transfer.yaml'))
    loaded = transfer_result['loaded_checkpoint']
    assert Path(loaded['path']) == expected_path
    assert loaded['epoch'] == selected_payload['epoch']
    assert loaded['metrics'] == selected_payload['metrics']
    assert loaded['context'] == selected_payload['context']


def test_stage_selection_inherits_guards_and_breaks_auc_ties_by_loss():
    selector = StageSelector.from_config({
        'default_selection': 'best_auc_with_loss_tolerance',
        'loss_tolerance_from_best_loss': .005,
        'max_allowed_val_loss': .404,
        'stage_specific': {'layer3': {'loss_tolerance_from_best_loss': .002}},
    }, 'layer3')
    assert selector.loss_tolerance == .002 and selector.absolute_limit == .404
    records = [
        {'epoch': 1, 'val_loss': .4, 'val_auc': .6},
        {'epoch': 2, 'val_loss': .4 + .002, 'val_auc': .7},
        {'epoch': 3, 'val_loss': .401, 'val_auc': .7},
        {'epoch': 4, 'val_loss': .403, 'val_auc': .9},
    ]
    assert selector.select(records, .4)['epoch'] == 3
    assert selector.select(records[:2], .4)['epoch'] == 2  # inclusive tolerance boundary
    capped = StageSelector('best_auc_with_loss_tolerance', .01, .4)
    assert capped.select(records, .4)['epoch'] == 1


@pytest.mark.parametrize('loss_config, weight', [
    ({'type': 'bce'}, 1.0),
    ({'type': 'bce_pos', 'pos_weight': 6.7615}, 6.7615),
])
def test_phase_validation_loss_matches_its_training_criterion(tmp_path, loss_config, weight):
    """Closed-form zero-logit loss distinguishes warm-up and full-data criteria."""
    import math
    from src.data.dataset import labels_from_dataset
    cfg = make_data(tmp_path)
    cfg['loss'] = loss_config
    loaders = runner.build_dataloaders(cfg)
    model = TinyBackbone()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    phase_trainer = trainer.Trainer(model, loaders['train'], loaders['val'], cfg, 'cpu')
    trained, validated = phase_trainer.run_epoch(optimizer)
    for partition, measured in [('train', trained), ('val', validated)]:
        labels = labels_from_dataset(loaders[partition].dataset)
        expected = math.log(2) * ((labels == 0).sum() + weight * labels.sum()) / len(labels)
        assert measured['loss'] == pytest.approx(expected, abs=1e-6)
