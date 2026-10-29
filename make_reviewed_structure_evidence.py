#!/usr/bin/env python3
"""Export manually inspected native structure crops without image-quality ranking."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from evaluate_visible_structure import read_frames, sha256_file, rgb, font


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--test_frames',type=Path,required=True)
    parser.add_argument('--source_images',type=Path,required=True)
    parser.add_argument('--ground_truth_dir',type=Path,required=True)
    parser.add_argument('--teacher_renders',type=Path,required=True)
    parser.add_argument('--model',action='append',required=True,help='LABEL=RENDERSDIR')
    parser.add_argument('--output_dir',type=Path,required=True)
    args=parser.parse_args(argv)
    frames=read_frames(args.test_frames)
    directories={'GT':args.ground_truth_dir,'teacher':args.teacher_renders}
    for spec in args.model:
        label,sep,path=spec.partition('=')
        if not sep or not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in label) or label in directories:
            raise ValueError('unique safe LABEL=RENDERSDIR required')
        directories[label]=Path(path)
    records=json.loads(args.reviews.read_text())
    labels=list(directories)
    width,height=290,225
    row_h=height+58
    sheet=Image.new('RGB',(len(labels)*width,75+len(records)*row_h),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((8,5),'Visually reviewed native SH3 examples: no PSNR ranking, same camera and same crop',fill='black',font=font(16))
    draw.text((8,28),'Near-disappearance can leave a diffuse remnant. PWSR20 is the original method, not the new structure-protected model.',fill='black',font=font(12))
    for col,label in enumerate(labels):draw.text((col*width+8,50),label,fill='black',font=font(17))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    native=args.output_dir/'native_crops';native.mkdir(exist_ok=True)
    manifests=[]
    nearest=getattr(Image,'Resampling',Image).NEAREST
    for row,record in enumerate(records):
        frame=Path(record['frame']).stem
        if frame not in frames:raise ValueError('review is not a held-out frame')
        index=frames.index(frame);filename='%05d.png'%index
        gt=rgb(args.ground_truth_dir/filename)
        matches=[p for p in args.source_images.iterdir() if p.stem==frame]
        if len(matches)!=1:raise ValueError('ambiguous source frame')
        source=rgb(matches[0])
        if source.shape!=gt.shape or np.max(np.abs(source-gt))>2./255.:raise ValueError('source/GT mapping mismatch')
        x0,y0,x1,y1=box=[int(i) for i in record['bbox_xyxy']]
        if not (0<=x0<x1<=gt.shape[1] and 0<=y0<y1<=gt.shape[0]):raise ValueError('invalid native bbox')
        margin=min(x0,y0,gt.shape[1]-x1,gt.shape[0]-y1)
        caption='{} | {} | native {}x{} | border margin {} px'.format(record['category'],frame,x1-x0,y1-y0,margin)
        y=75+row*row_h
        draw.rectangle((0,y,sheet.width-1,y+row_h-1),outline='#cccccc')
        draw.text((8,y+4),caption,fill='black',font=font(13))
        status=record['review_status']
        draw.text((8,y+25),status[:170],fill='black',font=font(11))
        saved={}
        for col,(label,directory) in enumerate(directories.items()):
            path=directory/filename
            im=Image.open(str(path)).convert('RGB')
            if im.size!=(gt.shape[1],gt.shape[0]):raise ValueError('all renders must use native GT dimensions')
            crop=im.crop(box)
            out=native/('{:02d}_{}_{}.png'.format(row,frame,label));crop.save(str(out))
            saved[label]=dict(source=str(path.resolve()),source_sha256=sha256_file(path),native_crop=str(out.resolve()))
            scale=min((width-12)/crop.width,(height-12)/crop.height)
            show=crop.resize((max(1,int(round(crop.width*scale))),max(1,int(round(crop.height*scale)))),nearest)
            sheet.paste(show,(col*width+(width-show.width)//2,y+58+(height-show.height)//2))
        manifests.append(dict(record,render_filename=filename,border_margin_px=margin,files=saved))
    out=args.output_dir/'reviewed_native_comparison.png';sheet.save(str(out))
    report=dict(protocol='fixed visually reviewed native SH3 bboxes, no PSNR ranking or image manipulation',
                source_reviews=str(args.reviews.resolve()),source_reviews_sha256=sha256_file(args.reviews),
                test_frames=frames,cases=manifests,contact_sheet_resampling='nearest',native_crops_resized=False,
                note='PWSR20 is the unmodified global-SPS control. Improved results over LG20 alone do not establish the contribution of the proposed extension.')
    (args.output_dir/'reviewed_native_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out.resolve()),cases=len(records),columns=labels),indent=2))


if __name__=='__main__':main()
