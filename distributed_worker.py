#!/usr/bin/env python3
import argparse,glob,math,os,shutil,subprocess,sys,time
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    from realesrgan.archs.rrdbnet_arch import RRDBNet
except Exception:
    from basicsr.archs.rrdbnet_arch import RRDBNet


MODELS={
    "anime_video":("realesr-animevideov3.pth","srvgg"),
    "anime_image":("RealESRGAN_x4plus_anime_6B.pth","rrdb"),
    "game":("realesr-general-x4v3.pth","srvgg"),
    "real":("RealESRGAN_x4plus.pth","rrdb")
}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--worker-id",type=int,required=True)
    p.add_argument("--workers",type=int,required=True)
    p.add_argument("--total-frames",type=int,required=True)
    p.add_argument("--fps",type=float,required=True)
    p.add_argument("--scale",type=float,default=2)
    p.add_argument("--model",default="anime_video")
    p.add_argument("--frame-start",type=int,required=True)
    p.add_argument("--frame-end",type=int,required=True)
    return p.parse_args()

def model_path(name):
    root=Path("weights")
    p=root/name
    if p.exists():
        return p
    p=Path(name)
    if p.exists():
        return p
    raise FileNotFoundError(f"Model not found: {name}")

def build_upscaler(model_name,scale):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    filename,arch=MODELS[model_name]
    path=model_path(filename)
    tile=256
    if arch=="srvgg":
        model=SRVGGNetCompact(
            num_in_ch=3,num_out_ch=3,num_feat=64,
            num_conv=32,num_out_ch=3,upscale=4,act_type="prelu"
        )
    else:
        model=RRDBNet(
            num_in_ch=3,num_out_ch=3,num_feat=64,
            num_block=6,num_grow_ch=32,scale=4
        )
    return RealESRGANer(
        scale=4,
        model_path=str(path),
        model=model,
        tile=tile,
        tile_pad=16,
        pre_pad=0,
        half=False,
        device=torch.device("cpu")
    )

def extract_range(src,start,end,dst,fps):
    Path(dst).mkdir(parents=True,exist_ok=True)
    pattern=str(Path(dst)/"input_%08d.png")
    vf=f"select='between(n\\,{start}\\,{end})'"
    cmd=[
        "ffmpeg","-y","-v","error",
        "-i",src,
        "-vf",vf,
        "-vsync","0",
        pattern
    ]
    subprocess.run(cmd,check=True)
    files=sorted(glob.glob(str(Path(dst)/"input_*.png")))
    expected=end-start+1
    if len(files)!=expected:
        raise RuntimeError(f"Extraction failed: {len(files)}/{expected}")
    return files

def upscale():
    a=args()
    out=Path(a.output)
    work=out/"_input"
    out.mkdir(parents=True,exist_ok=True)
    if out.exists():
        for p in out.glob("frame_*.png"):
            p.unlink()
    files=extract_range(
        a.input,
        a.frame_start,
        a.frame_end,
        work,
        a.fps
    )
    up=build_upscaler(a.model,a.scale)
    started=time.time()
    expected=len(files)
    for i,src in enumerate(files):
        frame_no=a.frame_start+i
        img=cv2.imread(src,cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read frame {src}")
        result,_=up.enhance(img,outscale=float(a.scale))
        target=out/f"frame_{frame_no:08d}.png"
        if not cv2.imwrite(str(target),result):
            raise RuntimeError(f"Cannot write {target}")
        done=i+1
        elapsed=time.time()-started
        rate=done/elapsed if elapsed else 0
        eta=(expected-done)/rate if rate else 0
        print(
            f"[WORKER {a.worker_id:02d}/{a.workers:02d}] "
            f"frame {done}/{expected} | "
            f"global {frame_no+1}/{a.total_frames} | "
            f"{rate:.2f} fps | ETA {eta:.1f}s",
           flush=True
        )
    produced=sorted(out.glob("frame_*.png"))
    if len(produced)!=expected:
        raise RuntimeError(f"Output mismatch {len(produced)}/{expected}")
    shutil.rmtree(work,ignore_errors=True)
    print(
        f"WORKER {a.worker_id} COMPLETE | "
        f"frames={expected} | time={time.time()-started:.1f}s",
       flush=True
    )

if __name__=="__main__":
    try:
        upscale()
    except Exception as e:
        print(f"WORKER FAILED: {e}",file=sys.stderr,flush=True)
        raise
