"""Serial GPU queue: frozen City evaluator, main comparison, seven ablations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from utils.distill_utils import atomic_json_save,sha256_file

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/city2_comparison_20261104'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    manifest=OUT/'benchmark_queue.json'
    if manifest.is_file():
        if not args.resume:raise FileExistsError(str(manifest))
        state=json.loads(manifest.read_text());state['status']='running'
    else:state=dict(status='running',completed=[],tasks=[])
    def run(name,script,*argv):
        if name in state['completed']:return
        command=[sys.executable,str(ROOT/script)]+list(map(str,argv))
        state['tasks'].append(dict(name=name,command=command,script_sha256=sha256_file(ROOT/script)));atomic_json_save(state,manifest)
        log=OUT/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            result=process.wait()
        if result:raise subprocess.CalledProcessError(result,command)
        state['completed'].append(name);atomic_json_save(state,manifest)
    try:
        render=OUT/'render_pipeline.json'
        while True:
            d=json.loads(render.read_text()) if render.is_file() else {}
            if d.get('status')=='complete' and d.get('visual_metrics_complete'):break
            print('Waiting for serial common-render job to finish',flush=True);time.sleep(10)
        run('semantic_train','city_semantic.py','train','--data_manifest',OUT/'semantic/dataset_manifest.json',
             '--output',OUT/'semantic/model','--steps',1500,'--batch_size',4,'--crop',384)
        run('prepare_main_semantics','prepare_semantic_comparison_inputs.py')
        run('infer_main_semantics','city_semantic.py','infer','--checkpoint',OUT/'semantic/model/model.pt',
            '--inputs_json',OUT/'semantic/inputs_main.json','--output',OUT/'semantic/predictions_main')
        run('evaluate_main_semantics','evaluate_semantic_comparison.py','--inputs',OUT/'semantic/inputs_main.json',
            '--predictions',OUT/'semantic/predictions_main/predictions_manifest.json','--output',OUT/'semantic/main_metrics.json')
        run('main_report','finalize_city_benchmark.py','--allow_pending')
        run('ablation_training','run_comparison_ablations.py','--run','--resume')
        run('prepare_all_semantics','prepare_semantic_comparison_inputs.py','--include_ablations')
        run('infer_all_semantics','city_semantic.py','infer','--checkpoint',OUT/'semantic/model/model.pt',
            '--inputs_json',OUT/'semantic/inputs_all.json','--output',OUT/'semantic/predictions_all')
        run('evaluate_all_semantics','evaluate_semantic_comparison.py','--inputs',OUT/'semantic/inputs_all.json',
            '--predictions',OUT/'semantic/predictions_all/predictions_manifest.json','--output',OUT/'semantic/all_metrics.json')
        run('final_report','finalize_city_benchmark.py')
        state['status']='complete'
    except BaseException as e:
        state.update(status='failed',error=str(e));atomic_json_save(state,manifest);raise
    atomic_json_save(state,manifest)

if __name__=='__main__':main()
