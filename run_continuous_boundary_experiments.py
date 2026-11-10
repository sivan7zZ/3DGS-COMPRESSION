"""Serial SH3-only 10x/20x continuous-boundary PUP experiments and comparison registry."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from utils.distill_utils import atomic_json_save, sha256_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/city2_continuous_boundary_20261109'
BINARY = ROOT / 'output/city2_boundary_weighted_20261106'
BASE = ROOT / 'output/city2_comparison_20261104'
PREVIOUS = ROOT / 'output/city2_pup_rir_dpra_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def snapshot():
    from run_pup_rir_experiments import TRAINING_FILES
    folder = OUT / 'source_snapshot'
    manifest = folder / 'manifest.json'
    if manifest.is_file():
        frozen = read(manifest)
        for name, digest in frozen['files'].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError('Frozen experiment source changed: ' + name)
        for key in ('sidecar', 'manifest'):
            row = frozen['continuous_priors'][key]
            if sha256_file(row['path']) != row['sha256']:
                raise ValueError('Frozen continuous prior changed: ' + key)
        return
    files = [name for name in TRAINING_FILES if name not in
             ('distill_train.py', 'configs/city_pup_rir_dpra_10x.json', 'configs/city_pup_rir_dpra_20x.json')]
    files += ['run_boundary_weighted_pipeline.py', 'run_continuous_boundary_experiments.py', 'run_boundary_weighted_experiments.py',
              'validate_boundary_weighted_results.py', 'validate_adaptive_results.py', 'validate_continuous_boundary_results.py',
              'prepare_continuous_boundary.py', 'utils/continuous_boundary_prior.py',
              'configs/city_continuous_boundary_10x.json', 'configs/city_continuous_boundary_20x.json']
    for path in (ROOT / 'submodules/pup-fisher').rglob('*'):
        if path.is_file() and not any(x in ('build', '__pycache__') for x in path.parts) and path.suffix in ('.cu', '.h', '.cpp', '.py', '.so'):
            files.append(str(path.relative_to(ROOT)))
    records = {}
    for name in sorted(set(files)):
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ROOT / name), str(destination))
        records[name] = sha256_file(ROOT / name)
    priors = OUT / 'continuous_priors'
    prior_files = {key: dict(path=str(path), sha256=sha256_file(path)) for key, path in (
        ('sidecar', priors / 'continuous_metadata.npz'), ('manifest', priors / 'metadata_manifest.json'))}
    if read(priors / 'metadata_manifest.json')['status'] != 'complete':
        raise ValueError('Continuous preprocessing is incomplete')
    atomic_json_save(dict(status='frozen', files=records, continuous_priors=prior_files, git_base='v3.2.0',
                         distillation=False, method='PUP_continuous_boundary_RIR_DPRA'), manifest)


def standard_metrics():
    import torch
    from lpipsPyTorch import LPIPS
    from evaluate_comparison_lpips import standard_input, implementation_provenance, state_dict_sha256
    criterion = LPIPS('vgg', '0.1').cuda().eval()
    criterion.requires_grad_(False)
    provenance = implementation_provenance(ROOT, BASE)
    provenance['state_dict_sha256'] = state_dict_sha256(criterion)
    provenance['source_files_sha256'][str(Path(__file__))] = sha256_file(__file__)
    original = next(row for row in read(BASE / 'models.json')['models'] if row['id'] == 'original_3dgs')
    names = ['%05d.png' % index for index in range(9)]
    records = {}
    for ratio in (10, 20):
        folder = OUT / ('full_%dx' % ratio) / 'stage3'
        stage = read(folder / 'run_args.json')
        method = 'ours_%d' % stage['final_iteration']
        test = folder / 'test' / method
        legacy = read(folder / 'results.json')[method]
        per_view, inputs = {}, []
        if sorted(path.name for path in (test / 'renders').glob('*.png')) != names:
            raise ValueError('Expected exactly nine held-out renders')
        for name in names:
            render = np.asarray(Image.open(test / 'renders' / name).convert('RGB'))
            target = np.asarray(Image.open(test / 'gt' / name).convert('RGB'))
            canonical = np.asarray(Image.open(Path(original['common_gt_dir']) / name).convert('RGB'))
            if render.shape != (894, 1592, 3) or not np.array_equal(target, canonical):
                raise ValueError('Native image/camera mismatch')
            a = torch.from_numpy(standard_input(render))[None].cuda()
            b = torch.from_numpy(standard_input(target))[None].cuda()
            with torch.no_grad():
                value = float(criterion(a, b).mean())
            if not np.isfinite(value):
                raise ValueError('Nonfinite LPIPS')
            per_view[name] = value
            inputs.append(dict(filename=name, render_sha256=sha256_file(test / 'renders' / name),
                               gt_sha256=sha256_file(test / 'gt' / name)))
        result = dict(status='complete', PSNR=legacy['PSNR'], SSIM=legacy['SSIM'],
                      LPIPS=float(np.mean(list(per_view.values()))), LPIPS_legacy_0_1=legacy['LPIPS'],
                      per_view_standard_lpips=per_view, criterion=provenance, input_range=[-1, 1], images=inputs)
        atomic_json_save(result, folder / 'metrics_standard.json')
        records['%dx_SH3' % ratio] = result
        print('STANDARD PNG', ratio, {key: result[key] for key in ('PSNR', 'SSIM', 'LPIPS')}, flush=True)
    atomic_json_save(dict(status='complete', results=records), OUT / 'metrics_standard.json')


def comparison_registry():
    registry = copy.deepcopy(read(BINARY / 'comparison/models.json'))
    for ratio in (10, 20):
        run = OUT / ('full_%dx' % ratio)
        stage = read(run / 'stage3/run_args.json')
        model_dir = run / 'stage3'
        registry['models'].append(dict(id='continuous_%dx' % ratio,
            method='PUP continuous SAM/depth/RGB boundary', target_ratio=ratio,
            actual_count=stage['final_gaussians'], model_dir=str(model_dir),
            ply=stage['last_ply'], checkpoint=stage['last_checkpoint'], iteration=stage['final_iteration'],
            common_render_dir=str(model_dir / 'test' / ('ours_%d' % stage['final_iteration']) / 'renders'),
            common_gt_dir=str(model_dir / 'test' / ('ours_%d' % stage['final_iteration']) / 'gt'), status='complete'))
    registry.update(output_root=str(OUT / 'comparison'),
                    semantic_protocol='Identical frozen five-fold City evaluators as 20261106; independent two held-out views and cross-fitted eight training-view auxiliaries separately')
    destination = OUT / 'comparison/models.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(registry, destination)
    return registry


def report():
    metrics = read(OUT / 'metrics_standard.json')['results']
    rows = []
    for ratio in (10, 20):
        validation = read(OUT / ('full_%dx' % ratio) / 'validation.json')
        values = metrics['%dx_SH3' % ratio]
        rows.append(dict(ratio=ratio, stage='SH3', gaussians=validation['target'],
                         **{key: values[key] for key in ('PSNR', 'SSIM', 'LPIPS')},
                         ply=validation['SH3']['ply'], checkpoint=validation['SH3']['checkpoint'],
                         ply_sha256=validation['SH3']['ply_sha256']))
    registry = comparison_registry()
    atomic_json_save(dict(status='complete', method='PUP_continuous_boundary_RIR_DPRA',
                         distillation=False, rows=rows), OUT / 'summary.json')
    lines = ['# City：SAM/深度/RGB连续边界PUP + RIR/DPRA（无蒸馏）', '',
             '只运行SH3压缩与评测；未运行SH蒸馏。实例与拓扑保护不变，边界只参与评分和恢复损失，不设边界剪枝/合并硬限制。', '',
             '评分为PUP六维Fisher logdet + log(1+5B)，B=0.4×SAM轮廓强度+0.4×深度边缘强度+0.2×RGB边缘强度。三通道均经原生alpha*T渲染贡献汇总至高斯，为[0,1]连续值；det重要性乘数范围[1,6]，奇异Fisher的−inf保持。',
             '合并上限为每轮输入高斯数的1%，回填预算为每轮目标数的2%；实际合并仍由安全相似候选决定。最终点数精确不变。', '',
             '| 倍率 | 高斯数 | PSNR↑ | SSIM↑ | 标准LPIPS↓ |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %d× | %d | %.4f | %.5f | %.5f |' %
                     (row['ratio'], row['gaussians'], row['PSNR'], row['SSIM'], row['LPIPS']))
    lines += ['', '视觉指标为9个固定独立测试视角、原生PNG；LPIPS为VGG0.1标准[-1,1]。语义指标使用匹配City标签，2个独立测试视角与8个训练视角辅助结果分列。', '',
              '比较模型共%d个（另加GT RGB）；LightGaussian10×比其他10×多1点，Speedy初始化不同。旧方法保留既有模型，所有语义统计须使用同一分折协议重算。' % len(registry['models']), '',
              '完整模型、checkpoint、逐轮raw/weighted评分、元数据、恢复状态及精确核验在full_10x/full_20x下。',
              '相对20261106二值边界版本，仅改变边界评分为连续三通道；实例、拓扑、恢复损失、合并1%/回填2%保持。权重为预先固定启发式，未用评测标签调参。']
    (OUT / 'README_压缩结果.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot()
    manifest = OUT / 'experiment_queue.json'
    if manifest.is_file():
        if not args.resume:
            raise FileExistsError('Use --resume for the existing queue')
        state = read(manifest)
        state['status'] = 'running'
    else:
        state = dict(status='running', completed=[], tasks=[], distillation=False)
    atomic_json_save(state, manifest)

    def completed(name):
        state['completed'].append(name)
        atomic_json_save(state, manifest)

    try:
        for ratio in (10, 20):
            snapshot()
            name = 'pipeline_%dx' % ratio
            if name not in state['completed']:
                command = [sys.executable, str(ROOT / 'run_boundary_weighted_pipeline.py'), '--config',
                           str(ROOT / ('configs/city_continuous_boundary_%dx.json' % ratio))]
                if args.resume:
                    command.append('--resume')
                state['tasks'].append(dict(name=name, command=command))
                atomic_json_save(state, manifest)
                with (OUT / (name + '.log')).open('a') as stream:
                    subprocess.check_call(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
                completed(name)
            name = 'validate_%dx' % ratio
            if name not in state['completed']:
                from validate_continuous_boundary_results import validate
                validate(OUT, ratio)
                completed(name)
        if 'standard_metrics' not in state['completed']:
            standard_metrics()
            completed('standard_metrics')
        snapshot()
        report()
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        atomic_json_save(state, manifest)
        raise
    atomic_json_save(state, manifest)


if __name__ == '__main__':
    main()
