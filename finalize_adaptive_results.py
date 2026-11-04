#!/usr/bin/env python3
"""CPU-only audit/report of adaptive compression and frozen SH3 crop comparisons.

No training, CUDA rendering, metric calculation, or test-driven model selection
is performed. Numerical quality values are read only from existing PNG-metric
artifacts. Fixed crop coordinates precede this experiment; visual conclusions
must be supplied separately after inspecting the generated comparisons.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from utils.structure_scene_io import read_ply,sha256_file

REPO=Path(__file__).resolve().parent
DEFAULT=REPO/'output/city2_adaptive_constraints_20261102'
BOXES=[('frame_024',(1321,30,1380,69),'small_region','Black parked vehicle'),
       ('frame_064',(1350,821,1373,844),'small_region','Small white garden object'),
       ('frame_056',(1352,70,1382,101),'boundary','Small rooftop-unit outline'),
       ('frame_008',(1183,77,1232,116),'thin_region','Narrow window-frame divisions'),
       ('frame_000',(157,31,247,70),'boundary','Garden fence and panel divisions'),
       ('frame_024',(83,192,111,225),'small_region','Red parked vehicle / reference control')]


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def model_record(directory,iteration,tests,check_render=True):
    directory=Path(directory);iteration=int(iteration)
    ply=directory/'point_cloud'/('iteration_%d'%iteration)/'point_cloud.ply'
    renders=directory/'test'/('ours_%d'%iteration)/'renders'
    result=dict(directory=str(directory.resolve()),iteration=iteration,ply=str(ply.resolve()),renders=str(renders.resolve()),status='pending',metrics=None)
    if not ply.is_file():return result
    vertices,_=read_ply(ply);names=vertices.dtype.names
    sh_rest=sum(k.startswith('f_rest_') for k in names);degree=int(round(math.sqrt(sh_rest/3.+1)-1))
    standard=('x','y','z','nx','ny','nz')+tuple('f_dc_%d'%i for i in range(3))+tuple('f_rest_%d'%i for i in range(sh_rest))+('opacity',)+tuple('scale_%d'%i for i in range(3))+tuple('rot_%d'%i for i in range(4))
    if tuple(names)!=standard or sh_rest!=3*((degree+1)**2-1):raise ValueError('Non-native Gaussian PLY schema: '+str(ply))
    result.update(gaussians=len(vertices),sh_degree=degree,property_count=len(names),ply_bytes=ply.stat().st_size,ply_sha256=sha256_file(ply),status='model_saved')
    metrics_file=directory/'results.json';key='ours_%d'%iteration
    if metrics_file.is_file() and key in read_json(metrics_file):
        metrics=read_json(metrics_file)[key]
        if any(not math.isfinite(float(metrics[k])) for k in ('PSNR','SSIM','LPIPS')):raise ValueError('Nonfinite metrics')
        result.update(metrics={k:float(metrics[k]) for k in ('PSNR','SSIM','LPIPS')},metrics_file=str(metrics_file.resolve()),metrics_sha256=sha256_file(metrics_file))
    images=sorted(renders.glob('*.png'));expected=['%05d.png'%i for i in range(len(tests))]
    result['render_count']=len(images)
    result['renders_complete']=[p.name for p in images]==expected
    if check_render and result['renders_complete']:
        result['render_sha256']={p.name:sha256_file(p) for p in images}
        for p in images:
            if Image.open(p).size!=(1592,894):raise ValueError('Non-native image dimensions: '+str(p))
    benchmark_path=directory/'render_benchmark.json'
    if benchmark_path.is_file():
        benchmark=read_json(benchmark_path)
        if benchmark.get('ply_sha256')!=result['ply_sha256']:raise ValueError('Render benchmark belongs to another PLY')
        result['render_benchmark']=dict(benchmark,artifact_sha256=sha256_file(benchmark_path))
    if result['metrics'] is not None and result['renders_complete']:result['status']='complete'
    return result


def baseline_records(ratio,tests):
    old=REPO/'output/city2_structure_guard_20261027'
    if ratio==10:
        return dict(Native=model_record(old/'reference/original',30000,tests),
                    LG=model_record(old/'reference/lightgaussian10',35000,tests),
                    PWSR=model_record(REPO/'output/city2_instance_guard_20261026/control/stage3',25199,tests),
                    v3=model_record(REPO/'output/city2_structure_guard_10x_20261030/stage3',25199,tests))
    return dict(Native=model_record(old/'reference/original',30000,tests),
                LG=model_record(old/'lightgaussian20/stage3',35000,tests),
                PWSR=model_record(old/'control20/stage3',25199,tests),
                v3=model_record(old/'protected20_adaptive/stage3',25199,tests))


def new_run_record(root,ratio,tests,allow_pending=False):
    run=root/('full_%dx'%ratio);stage=run/'stage3';args_path=stage/'run_args.json'
    result=dict(run=str(run.resolve()),status='pending',SH3=None,SH2=None)
    if not args_path.is_file():return result
    args=read_json(args_path);result['training_status']=args.get('status')
    if [Path(f).stem for f in args.get('test_views',[])]!=tests:raise ValueError('New model held-out camera order differs')
    result['config']=args.get('config',{})
    result['rounds']=[{k:e.get(k) for k in ('round_id','before','after_pruning','after_merging','temporary_count','after_reallocation','target','pruned','merged','restored','recovery_steps','total_steps','recovery_trigger','adaptation','quota_audit')} for e in args.get('events',[])]
    if 'final_iteration' not in args:return result
    iteration=args['final_iteration'];sh3=model_record(stage,iteration,tests)
    result['SH3']=sh3;result['completed_recovery_steps']=args.get('completed_steps');result['elapsed_compression_seconds']=args.get('elapsed_seconds')
    if sh3['status']!='pending':
        expected=int(args['target'])
        if args.get('status')=='complete' and sh3['gaussians']!=expected:raise ValueError('Exact target mismatch')
        metadata=stage/'compression_metadata.npz'
        if not metadata.is_file():raise FileNotFoundError(str(metadata))
        with np.load(metadata) as d:
            required=('instance_ids','confidence','original_ids','instance_fragility','boundary_mask','structure_risk','visibility')
            if any(len(d[k])!=sh3['gaussians'] for k in required):raise ValueError('Compressed metadata row mismatch')
            stats=dict(rows=sh3['gaussians'],known_instance_gaussians=int(np.count_nonzero(d['instance_ids']>=0)),unknown_instance_gaussians=int(np.count_nonzero(d['instance_ids']<0)),boundary_gaussians=int(np.count_nonzero(d['boundary_mask'])),source_lineage_rows=int(np.count_nonzero(d['original_ids']>=0)),merged_or_unresolved_lineage_rows=int(np.count_nonzero(d['original_ids']<0)))
        result['metadata_audit']=dict(stats,path=str(metadata.resolve()),sha256=sha256_file(metadata))
        checkpoint=Path(args['last_checkpoint'])
        if not checkpoint.is_file():raise FileNotFoundError(str(checkpoint))
        result['checkpoint']=str(checkpoint.resolve());result['quota_audit']=args.get('quota_audit')
    distill_args=run/'sh2/run_args.json'
    if distill_args.is_file():
        dargs=read_json(distill_args);sh2_iteration=int(dargs.get('final_iteration',35000+args.get('config',{}).get('distill_steps',5000)))
        sh2=model_record(run/'sh2',sh2_iteration,tests);result['SH2']=sh2
        if sh2.get('gaussians') is not None and sh3.get('gaussians')!=sh2['gaussians']:raise ValueError('SH2 changed Gaussian count')
        result['distillation_status']=dargs.get('status');result['distillation_steps']=dargs.get('completed_steps',dargs.get('distill_steps'))
        result['teacher_unchanged']=dargs.get('teacher_unchanged') is True
        result['distillation_audit_complete']=(dargs.get('status')=='complete' and result['distillation_steps']==int(args.get('config',{}).get('distill_steps',5000)) and result['teacher_unchanged'] and sh2.get('sh_degree')==2)
        if result['distillation_audit_complete']:
            sh2_checkpoint=Path(dargs['last_checkpoint'])
            if not sh2_checkpoint.is_file() or not Path(str(sh2_checkpoint)+'.state.pth').is_file():raise FileNotFoundError('SH2 checkpoint/optimizer sidecar missing')
        sidecar=run/'sh2/compression_metadata.npz'
        if sidecar.is_file() and sidecar.read_bytes()!=(stage/'compression_metadata.npz').read_bytes():raise ValueError('SH2 row metadata differs from SH3')
        result['SH2_metadata_present']=sidecar.is_file()
    pipeline_path=run/'pipeline.json'
    result['pipeline_status']=read_json(pipeline_path).get('status') if pipeline_path.is_file() else 'unknown'
    result['status']='complete' if result['pipeline_status']=='complete' and sh3['status']=='complete' and result.get('SH2') and result['SH2']['status']=='complete' and result.get('distillation_audit_complete') and result.get('SH2_metadata_present') else 'pending'
    return result


def comparison_sheet(ratio,models,tests,output):
    output.mkdir(parents=True,exist_ok=True);native=output/'native_crops';native.mkdir(exist_ok=True)
    teacher=models['Native'];gt=Path(teacher['directory'])/'test'/('ours_%d'%teacher['iteration'])/'gt'
    dirs=dict(GT=gt,Native=Path(teacher['renders']),LG=Path(models['LG']['renders']),PWSR=Path(models['PWSR']['renders']),v3=Path(models['v3']['renders']),Adaptive=Path(models['Adaptive']['renders']))
    width,plotheight,rowheight=290,200,250
    sheet=Image.new('RGB',(len(dirs)*width,70+len(BOXES)*rowheight),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,5),'Native SH3 / %dx Gaussian-count compression / fixed reference crops'%ratio,fill='black',font=font(16))
    draw.text((8,28),'Same cameras and pixels. No object disappearance or rescue is assumed. Zoom uses nearest-neighbour only.',fill='black',font=font(12))
    for column,label in enumerate(dirs):draw.text((column*width+8,49),label,fill='black',font=font(16))
    cases=[];nearest=getattr(Image,'Resampling',Image).NEAREST
    for row,(frame,box,category,description) in enumerate(BOXES):
        filename='%05d.png'%tests.index(frame);y=70+row*rowheight
        gt_array=np.asarray(Image.open(gt/filename).convert('RGB'),np.int16)
        matches=list(Path('/home/zhangys/gaussian-splatting（原生）/data/city/images').glob(frame+'.*'))
        if len(matches)!=1 or np.max(np.abs(np.asarray(Image.open(matches[0]).convert('RGB'),np.int16)-gt_array))>2:raise ValueError('Source frame/GT mismatch')
        draw.rectangle((0,y,sheet.width-1,y+rowheight-1),outline='#cccccc')
        draw.text((8,y+4),'%s | %s | %s | native %dx%d'%(frame,category,description,box[2]-box[0],box[3]-box[1]),fill='black',font=font(13))
        record=dict(frame=frame,bbox_xyxy=list(box),category=category,description=description,render_filename=filename,files={})
        for col,(label,directory) in enumerate(dirs.items()):
            path=directory/filename;image=Image.open(path).convert('RGB')
            if image.size!=(1592,894):raise ValueError('Native view size mismatch')
            if label!='GT':
                other_gt=directory.parent/'gt'/filename
                if not other_gt.is_file() or np.max(np.abs(np.asarray(Image.open(other_gt).convert('RGB'),np.int16)-gt_array))>2:raise ValueError('Model camera/GT order mismatch: '+label)
            crop=image.crop(box);save=native/('%02d_%s_%s.png'%(row,frame,label));crop.save(save)
            # Saved native crop is exact; overview magnification is explicit.
            if not np.array_equal(np.asarray(Image.open(save)),np.asarray(image)[box[1]:box[3],box[0]:box[2]]):raise AssertionError('Crop changed pixels')
            scale=min((width-12)/crop.width,(plotheight-12)/crop.height)
            zoom=crop.resize((int(round(crop.width*scale)),int(round(crop.height*scale))),nearest)
            sheet.paste(zoom,(col*width+(width-zoom.width)//2,y+42+(plotheight-zoom.height)//2))
            record['files'][label]=dict(source=str(path.resolve()),source_sha256=sha256_file(path),crop=str(save.resolve()),native_pixels_exact=True)
        cases.append(record)
    destination=output/'native_sh3_comparison.png';sheet.save(destination)
    write_json(output/'crop_manifest.json',dict(ratio=ratio,cases=cases,protocol='fixed six boxes before the experiment; no PSNR ranking; original pixels; nearest-neighbour contact sheet',manual_conclusion='pending visual inspection',labels=list(dirs)))
    return str(destination.resolve())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--ratios',type=int,nargs='+',choices=(10,20),default=[10,20]);p.add_argument('--allow_pending',action='store_true')
    p.add_argument('--manual_review',type=Path)
    args=p.parse_args(argv);root=args.root;report=root/'reports';report.mkdir(parents=True,exist_ok=True)
    shared=read_json(root/'shared/metadata_manifest.json');tests=list(shared['test_frames_excluded'])
    if tests!=sorted(tests) or len(tests)!=9:raise ValueError('Fixed nine held-out cameras required')
    summary=dict(schema_version=1,experiment_root=str(root.resolve()),evaluation='existing native PNG metrics only; SH3 primary comparison; SH2 separate',train_views=shared['train_frames'],test_views=tests,ratios={},manual_review={'status':'pending','conclusion':'尚未完成本次渲染图的目视检查；不预设地物消失或挽救。'})
    if args.manual_review:
        review=read_json(args.manual_review)
        if review.get('status') not in ('pending','complete'):raise ValueError('Invalid visual review status')
        summary['manual_review']=review
    for ratio in args.ratios:
        new=new_run_record(root,ratio,tests,args.allow_pending);baselines=baseline_records(ratio,tests)
        row=dict(new=new,baselines=baselines)
        visual=summary['manual_review'].get('ratios',{}).get(str(ratio))
        if visual and visual.get('status')=='complete' and visual.get('new_sh3_ply_sha256')!=new.get('SH3',{}).get('ply_sha256'):raise ValueError('Visual review belongs to another model')
        if new.get('SH3') and new['SH3']['renders_complete'] and all(v.get('renders_complete') for v in baselines.values()):
            row['comparison_sheet']=comparison_sheet(ratio,dict(baselines,Adaptive=new['SH3']),tests,report/('%dx_sh3'%ratio))
        if not args.allow_pending and new['status']!='complete':raise ValueError('%dx pipeline/metrics incomplete; use --allow_pending to report explicitly'%ratio)
        summary['ratios'][str(ratio)]=row
        if new.get('rounds'):
            with (report/('%dx_rounds.csv'%ratio)).open('w',newline='') as stream:
                keys=('round_id','before','after_pruning','after_merging','temporary_count','after_reallocation','target','pruned','merged','restored','recovery_steps','total_steps','recovery_trigger')
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows({k:r.get(k) for k in keys} for r in new['rounds'])
    write_json(report/'summary.json',summary)
    lines=['# 自适应约束压缩实验：10×与20×','',
           '模型采用全局 SPS 分数及排序，新增实例生存与边界配额、细长结构图约束、自适应剪枝恢复、受约束合并和困难区域预算重分配。所有数值读取实际保存的原生 PNG 评估；蒸馏前 SH3 是主要对比。','',
           '训练使用61个视角；9个固定测试视角仅用于渲染评估。标准 PLY 仅含3DGS属性，实例与结构信息另存 compression_metadata.npz。','',
           '| 倍率 | 模型/阶段 | 高斯数 | PLY MiB | PSNR | SSIM | LPIPS | FPS | 状态 |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for ratio in args.ratios:
        row=summary['ratios'][str(ratio)]
        entries=[(k,row['baselines'][k]) for k in ('Native','LG','PWSR','v3')]+[('本次 SH3',row['new'].get('SH3')),('本次 SH2',row['new'].get('SH2'))]
        for name,record in entries:
            if not record:lines.append('| %d× | %s | — | — | — | — | — | — | pending |'%(ratio,name));continue
            m=record.get('metrics');metrics='| %.4f | %.5f | %.5f |'%(m['PSNR'],m['SSIM'],m['LPIPS']) if m else '| — | — | — |'
            fps='%.1f'%record['render_benchmark']['fps'] if record.get('render_benchmark') else '—'
            lines.append('| %d× | %s | %s | %s %s %s | %s |'%(ratio,name,record.get('gaussians','—'),'%.2f'% (record['ply_bytes']/1024**2) if 'ply_bytes' in record else '—',metrics,fps,record['status']))
    lines+=['','倍率指相对原始1,788,093点的高斯数量压缩；SH蒸馏进一步降低每点存储。LightGaussian10的历史结果有178,810点，其余10×目标为178,809点；20×目标为89,405点。','',
           'FPS读取已有原生渲染计时：RTX2080Ti、1592×894、18帧预热后90帧，CUDA事件逐帧同步；不含加载与PNG写盘。未计时的历史模型显示—，不能据此作速度优劣结论。','']
    for ratio in args.ratios:
        row=summary['ratios'][str(ratio)];run=row['new'];lines+=['## %d×结果'%ratio,'','流水线状态：`%s`。恢复更新：`%s`；实际压缩轮数：`%d`。'%(run.get('pipeline_status','pending'),run.get('completed_recovery_steps','pending'),len(run.get('rounds',[]))),'']
        if row.get('comparison_sheet'):lines+=['![固定六例原生SH3对比](%s)'%row['comparison_sheet'],'']
        visual=summary['manual_review'].get('ratios',{}).get(str(ratio),{})
        if visual.get('conclusion'):lines+=[visual['conclusion'],'']
        for stage in ('SH3','SH2'):
            if run.get(stage) and run[stage].get('gaussians'):lines+=['- [%s标准PLY](%s)'%(stage,run[stage]['ply'])]
        lines+=['']
    lines+=['## 证据边界','',str(summary['manual_review'].get('conclusion',summary['manual_review'].get('overall_conclusion','目视结论见审查记录。'))),'',
            '六处裁剪坐标沿用实验前已固定的位置；不会按照本次PSNR或优势重新挑选。SAM是无类别伪实例，实例配额与稀疏图约束并不等于物体识别真值或完整三维拓扑保证。与LG的差异需要同时参考同预算v3和原PWSR结果。','',
            '元数据构建和近似说明见 [复现审计回执](%s)。'%(root/'shared/metadata_reproduction_receipt.json'),'逐例目视记录见 [人工审查记录](%s)。'%(root/'reports/manual_visual_review.json'),'']
    (root/'README_自适应压缩结果.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str((root/'README_自适应压缩结果.md').resolve()),statuses={r:x['new']['status'] for r,x in summary['ratios'].items()}),indent=2))


if __name__=='__main__':main()
