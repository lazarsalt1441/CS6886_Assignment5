"""Strict stdlib-only output checker; suitable for macOS's Python."""
from array import array
from pathlib import Path
import math,struct,sys

def load(path):
    raw=Path(path).read_bytes()
    if len(raw)<8:raise ValueError('truncated header')
    n,d=struct.unpack('<ii',raw[:8])
    if n<=0 or d<=0 or len(raw)!=8+4*n*d:raise ValueError('bad dimensions/file length')
    a=array('f');a.frombytes(raw[8:])
    if sys.byteorder!='little':a.byteswap()
    if not all(math.isfinite(v) for v in a):raise ValueError('non-finite output')
    return (n,d),a

def relative_error(ref,out):
    if len(ref)!=len(out):return float('inf')
    absolute=max(abs(a-b) for a,b in zip(ref,out));scale=max(abs(a) for a in ref)
    return absolute/scale if scale else (0 if absolute==0 else float('inf'))

def check(input_dir=Path('input'),output_dir=Path('output')):
    inputs=sorted(input_dir.glob('*.bin'))
    if not inputs:raise RuntimeError('no inputs')
    for inp in inputs:
        shape,ref=load(output_dir/('naive_'+inp.name))
        if shape!=struct.unpack('<ii',inp.read_bytes()[:8]):raise RuntimeError('wrong naive header')
        for variant in ['loops','tiled','fused','best','omp']:
            out_shape,out=load(output_dir/(variant+'_'+inp.name))
            err=relative_error(ref,out)
            if out_shape!=shape or err>1e-4:raise RuntimeError(f'{variant}: {inp.name}: error {err}')
            print(f'{inp.name} {variant} PASS error={err:.8g}',flush=True)

if __name__=='__main__':check()
