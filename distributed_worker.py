#!/usr/bin/env python3
import json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np, requests, torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
try:
    from realesrgan.archs.rrdbnet_arch import RRDBNet
except ModuleNotFoundError:
    from basicsr.archs.rrdbnet_arch import RRDBNet

CPU=max(1,os.cpu_count() or 1); torch.set_num_threads(CPU); torch.set_num_interop_threads(max(1,min(4,CPU)))
ROOT=Path('.'); W=ROOT/'weights'; WORK=ROOT/'worker_work'; OUT=ROOT/'worker_output'; WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
MODELS={
'anime_video':('realesr-animevideov3.pth','srvgg'),'anime_image':('RealESRGAN_x4plus_anime_6B.pth','rrdb'),'game':('realesr-general-x4v3.pth','srvgg'),'real':('RealESRGAN_x4plus.pth','rrdb')}

def run(cmd,**kw):
    print('+',' '.join(map(str,cmd)),flush=True); return subprocess.run(cmd,check=True,**kw)
def probe(p):
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-print_format','json','-show_streams','-show_format',str(p)],text=True)); v=next(x for x in d['streams'] if x.get('codec_type')=='video'); r=v.get('avg_frame_rate') or v.get('r_frame_rate') or '30/1'; a,b=r.split('/'); fps=float(a)/float(b) if float(b) else 30.; n=v.get('nb_frames'); frames=0
    try: frames=int(float(n)) if n not in (None,'','N/A') else 0
    except: frames=0
    if frames<=0: frames=max(1,round(float(v.get('duration') or d.get('format',{}).get('duration') or 0)*fps))
    return {'w':int(v['width']),'h':int(v['height']),'fps':fps,'frames':int(frames),'audio':any(x.get('codec_type')=='audio' for x in d['streams'])}
def model(key):
    f,arch=MODELS[key]; path=W/f
    if not path.is_file(): raise RuntimeError('missing model '+str(path))
    if arch=='srvgg':
        net=SRVGGNetCompact(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=32,upscale=4,act_type='prelu')
    else:
        net=RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
    return RealESRGANer(scale=4,model_path=str(path),model=net,tile=int(os.getenv('TILE','0') or 0),tile_pad=10,pre_pad=0,half=False)
def tg_download(file_id,token,out):
    r=requests.get(f'https://api.telegram.org/bot{token}/getFile',params={'file_id':file_id},timeout=30); r.raise_for_status(); p=r.json()['result']['file_path']; u=f'https://api.telegram.org/file/bot{token}/{p}';
    with requests.get(u,stream=True,timeout=120) as q:
        q.raise_for_status();
        with open(out,'wb') as f:
            for c in q.iter_content(1024*1024):
                if c:f.write(c)
def main():
    idx=int(os.environ['WORKER_INDEX']); workers=int(os.environ.get('TOTAL_WORKERS','20')); job=os.environ['JOB_ID']; src=WORK/'source'; filename=os.environ.get('FILENAME','input.mp4'); src.parent.mkdir(exist_ok=True)
    if idx==0 or True:
        print(f'WORKER {idx+1}/{workers} | CPU {CPU} | full throttle',flush=True)
    tg_download(os.environ['TG_FILE_ID'],os.environ['TG_BOT_TOKEN'],src)
    info=probe(src); total=min(info['frames'],int(os.environ.get('MAX_FRAMES','3600'))); base=total//workers; rem=total%workers; start=idx*base+min(idx,rem); count=base+(1 if idx<rem else 0); end=start+count-1
    if count<=0: raise RuntimeError(f'empty partition {idx}')
    key=os.environ.get('MODEL_KEY','anime_video'); scale=float(os.environ.get('SCALE','2')); outw=int(info['w']*scale)//2*2; outh=int(info['h']*scale)//2*2
    preset={'fast':('23','veryfast'),'balanced':('19','veryfast'),'best':('16','slow')}.get(os.environ.get('PRESET','balanced'),('19','veryfast')); wd=WORK/f'w{idx:02d}'; shutil.rmtree(wd,ignore_errors=True); wd.mkdir(parents=True)
    frames=wd/'frames'; frames.mkdir(); pattern=str(frames/'f_%08d.png')
    vf=f"select=between(n\\,{start}\\,{end}),setpts=N/FRAME_RATE/TB"; run(['ffmpeg','-y','-v','error','-i',str(src),'-an','-vf',vf,'-vsync','0',pattern])
    imgs=sorted(frames.glob('f_*.png')); expected=count
    if len(imgs)!=expected: raise RuntimeError(f'partition {idx}: expected {expected} frames, got {len(imgs)}')
    up=model(key); t=time.time();
    for n,p in enumerate(imgs,1):
        img=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError('bad frame '+str(p))
        try: arr=up.enhance(img,outscale=scale)[0]
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower(): raise
            up.tile=256; arr=up.enhance(img,outscale=scale)[0]
        if arr.shape[1]!=outw or arr.shape[0]!=outh: arr=cv2.resize(arr,(outw,outh),interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(p),arr)
        if n==1 or n%10==0 or n==expected: print(f'W{idx:02d} {n}/{expected} {n/max(0.001,time.time()-t):.2f} fps',flush=True)
    chunk=OUT/f'worker_{idx:02d}.mp4'; run(['ffmpeg','-y','-v','error','-framerate',f"{info['fps']:.8f}",'-i',pattern,'-c:v','libx264','-preset',preset[1],'-crf',preset[0],'-threads',str(CPU),'-pix_fmt','yuv420p','-movflags','+faststart',str(chunk)])
    manifest={'worker':idx,'workers':workers,'start':start,'end':end,'frames':expected,'fps':info['fps'],'width':outw,'height':outh,'filename':filename,'job_id':job}
    (OUT/f'worker_{idx:02d}.json').write_text(json.dumps(manifest,indent=2)); print('DONE',json.dumps(manifest),flush=True)
if __name__=='__main__': main()
