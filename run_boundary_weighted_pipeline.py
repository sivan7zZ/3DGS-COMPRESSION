"""SH3-only compression, native rendering and visual metrics; no distillation stage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from utils.checkpoint_utils import atomic_json_save

ROOT = Path(__file__).resolve().parent


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def commands(config_path, config, resume=False, smoke_rounds=0):
    """One SH3 model is produced. No teacher or SH2 dependency is accepted."""
    if config.get('pruning_score') not in ('pup_boundary_weighted', 'pup_continuous_boundary'):
        raise ValueError('Expected a supported boundary-weighted PUP configuration')
    if 'teacher_checkpoint' in config or 'distill_steps' in config:
        raise ValueError('SH3-only configuration must not contain distillation settings')
    output = resolve(config['output_root']).resolve()
    if (ROOT / 'output').resolve() not in output.parents:
        raise ValueError('Output must be inside repository output/')
    initial = ['train_pup_rir_compression.py', '--config', resolve(config_path)]
    if resume and (output / 'stage3/compression_state.pt').is_file():
        initial.append('--resume')
    if smoke_rounds:
        initial += ['--smoke_rounds', smoke_rounds]
    return initial, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_rounds', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text())
    initial, output = commands(config_path, config, args.resume, args.smoke_rounds)
    if args.dry_run:
        print(json.dumps(dict(compression=list(map(str, initial)), output=str(output),
                              stages=['compress', 'render_stage3', 'metrics'], distillation=False), indent=2))
        return
    manifest_path = output / 'pipeline.json'
    if manifest_path.is_file() and not args.resume:
        raise FileExistsError('Existing experiment requires --resume or a new output_root')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', config=config, completed=[], commands=[],
                    resume=args.resume, distillation=False)
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config or manifest.get('distillation') is not False:
            raise ValueError('Resume configuration or pipeline differs')
        manifest.update(status='running', resume=True)
    atomic_json_save(config, output / 'config.json')

    def run(name, argv):
        if name in manifest['completed']:
            return
        command = [sys.executable, str(ROOT / argv[0])] + list(map(str, argv[1:]))
        manifest['commands'].append(dict(name=name, command=command))
        atomic_json_save(manifest, manifest_path)
        with (output / (name + '.log')).open('a') as log:
            result = subprocess.call(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        if result:
            raise subprocess.CalledProcessError(result, command)
        manifest['completed'].append(name)
        atomic_json_save(manifest, manifest_path)

    try:
        run('compress', initial)
        stage = json.loads((output / 'stage3/run_args.json').read_text())
        if stage['status'] != 'complete' and not args.smoke_rounds:
            raise ValueError('Compression incomplete')
        common = ['-s', resolve(config['source_path']), '--eval', '--resolution', config.get('resolution', -1)]
        run('render_stage3', ['render.py'] + common + ['-m', output / 'stage3',
            '--iteration', stage['final_iteration'], '--skip_train'])
        run('metrics', ['metrics.py', '-m', output / 'stage3', '--device', 'cuda'])
        manifest['status'] = 'smoke_complete' if args.smoke_rounds else 'complete'
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        atomic_json_save(manifest, manifest_path)
        raise
    atomic_json_save(manifest, manifest_path)


if __name__ == '__main__':
    main()
