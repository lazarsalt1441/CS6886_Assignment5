"""Independent stable FP64 reference; no Python packages required."""
from pathlib import Path
from array import array
import json,math,os,random,struct,subprocess,tempfile
from check_outputs import load,relative_error
ROOT=Path(__file__).resolve().parents[1]
CASES=[(1,1,1,'gaussian'),(3,7,1,'gaussian'),(17,33,1,'gaussian'),
       (31,64,1,'gaussian'),(33,65,1,'gaussian'),(65,128,1,'gaussian'),
       (67,129,1,'gaussian'),(96,64,20,'gaussian'),(33,65,0,'gaussian'),
       (47,17,1,'gaussian'),(48,64,1,'gaussian'),(49,65,1,'gaussian'),
       (127,33,1,'gaussian'),(128,64,1,'gaussian'),(129,65,1,'gaussian'),
       (257,64,1,'gaussian'),(257,64,20,'gaussian'),
       (385,7,1,'later_maximum'),(385,7,1,'earlier_maximum'),
       (257,9,1,'constant_value'),(257,9,1,'zero_value')]

def run_edges(binaries=None,environment=None,prefix=None,evidence=None):
    binaries=binaries or {v:ROOT/'build'/v for v in ['naive','loops','tiled','fused','best','omp']}
    prefix=prefix or [];environment=environment or os.environ.copy()
    rng=random.Random(6886);records=[]
    with tempfile.TemporaryDirectory() as td:
      for n,d,amp,kind in CASES:
        # Round inputs to FP32 before computing the independent FP64 reference.
        q,k,v=[list(array('f',(rng.gauss(0,1)*a for _ in range(n*d)))) for a in [amp,amp,1]]
        if kind in ['later_maximum','earlier_maximum']:
            q=[0.0]*(n*d);k=[0.0]*(n*d)
            sign=1 if kind=='later_maximum' else -1
            for i in range(n):q[i*d]=float(array('f',[math.sqrt(d)])[0])
            for j in range(n):k[j*d]=float(sign*25*(j//128-1))
        if kind=='constant_value':v=[(-1.0 if c%2 else 1.0)*(c+1)/d for _ in range(n) for c in range(d)]
        if kind=='zero_value':v=[0.0]*(n*d)
        # Structured cases must use the same FP32 inputs as the C driver.
        q,k,v=[list(array('f',x)) for x in [q,k,v]]
        ref=[]
        for i in range(n):
            score=[sum(q[i*d+l]*k[j*d+l] for l in range(d))/math.sqrt(d) for j in range(n)]
            mx=max(score);weights=[math.exp(x-mx) for x in score];total=sum(weights)
            ref.extend(sum(weights[j]*v[j*d+l] for j in range(n))/total for l in range(d))
        inp=Path(td)/'input.bin';out=Path(td)/'output.bin'
        inp.write_bytes(struct.pack('<ii',n,d)+b''.join(struct.pack('<%df'%len(x),*x) for x in [q,k,v]))
        for variant,binary in binaries.items():
          for threads in ([1,2,4] if variant=='omp' else [1]):
            env=environment.copy();env.update(OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE')
            subprocess.run([*prefix,str(binary),str(inp),str(out)],check=True,env=env,stdout=subprocess.DEVNULL)
            shape,result=load(out);err=relative_error(ref,result)
            if shape!=(n,d) or err>1e-4:raise RuntimeError(f'{variant} N={n} d={d} t={threads} error={err}')
            records.append(dict(variant=variant,N=n,d=d,amplitude=amp,case=kind,threads=threads,relative_error_fp64=err))
    if evidence:Path(evidence).write_text(json.dumps(records,indent=2))
    print(f'PASS: {len(records)} independent FP64 checks; max error {max(r["relative_error_fp64"] for r in records):.8g}',flush=True)
    return records

if __name__=='__main__':run_edges(evidence=ROOT/'src/evidence/edge_checks.json')
