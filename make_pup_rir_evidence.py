"""Exact fixed-coordinate native crops: GT, v3.1, new PUP/RIR and PWSR."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from finalize_city_benchmark import BOXES,font,NEAREST,read,write,sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/city2_pup_rir_dpra_20261105'
OLD=ROOT/'output/city2_comparison_20261104'


def make(ratio):
    registry=read(OLD/'models.json');byid={x['id']:x for x in registry['models']};frames=registry['test_frames']
    old=byid['ours_%dx'%ratio];control=byid['pwsr_%dx'%ratio]
    new=OUT/('full_%dx'%ratio)/'stage3/test/ours_25199'
    columns=[('GT',Path(old['common_gt_dir'])),('v3.1.0',Path(old['common_render_dir'])),
             ('PUP + RIR/DPRA',new/'renders'),('PWSR',Path(control['common_render_dir']))]
    output=OUT/'reports'/('direct_v310_%dx'%ratio);(output/'native_crops').mkdir(parents=True,exist_ok=True)
    width,height=330,250
    sheet=Image.new('RGB',(width*len(columns),70+height*len(BOXES)),'white');d=ImageDraw.Draw(sheet)
    d.text((8,3),'%dx SH3 / same six coordinates / exact native crops, nearest-neighbour overview'%ratio,fill='black',font=font(16))
    d.text((8,25),'Locations fixed before this run; presence/rescue requires visual review.',fill='black',font=font(13))
    for j,(label,_) in enumerate(columns):d.text((j*width+8,47),label,fill='black',font=font(17))
    cases=[]
    for i,(frame,box,category,description) in enumerate(BOXES):
        name='%05d.png'%frames.index(frame);y=70+i*height
        d.rectangle((0,y,sheet.width-1,y+height-1),outline='#cccccc')
        d.text((8,y+4),'%s | %s | %dx%d native pixels'%(frame,description,box[2]-box[0],box[3]-box[1]),fill='black',font=font(13))
        reference=np.asarray(Image.open(columns[0][1]/name).convert('RGB'));records={}
        for j,(label,folder) in enumerate(columns):
            path=folder/name;image=Image.open(path).convert('RGB')
            if image.size!=(1592,894):raise ValueError('Non-native geometry')
            if j and not np.array_equal(np.asarray(Image.open(folder.parent/'gt'/name).convert('RGB')),reference):raise ValueError('GT mismatch')
            crop=image.crop(box);target=output/'native_crops'/('%02d_%s_%d.png'%(i,frame,j));crop.save(target)
            if not np.array_equal(np.asarray(crop),np.asarray(image)[box[1]:box[3],box[0]:box[2]]):raise ValueError('Native crop differs')
            factor=min((width-12)/crop.width,188./crop.height)
            zoom=crop.resize((round(crop.width*factor),round(crop.height*factor)),NEAREST)
            sheet.paste(zoom,(j*width+(width-zoom.width)//2,y+42+(200-zoom.height)//2))
            records[label]=dict(source=str(path),source_sha256=sha(path),crop=str(target),crop_sha256=sha(target),native_exact=True)
        cases.append(dict(frame=frame,bbox_xyxy=box,category=category,description=description,files=records))
    target=output/'native_comparison.png';sheet.save(target)
    write(output/'manifest.json',dict(ratio=ratio,columns=[x[0] for x in columns],cases=cases,figure=str(target),figure_sha256=sha(target)))
    print(str(target))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ratio',choices=[10,20],type=int,nargs='+',default=[10,20]);a=p.parse_args()
    for ratio in a.ratio:make(ratio)
