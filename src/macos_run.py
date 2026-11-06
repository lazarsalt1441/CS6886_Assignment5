"""Native M1 measurement bundle. Uses only Python's standard library.
No Linux taskset, /proc, /sys, pip packages, sudo or Mac-specific private APIs.
Does not manufacture Mac measurements from the earlier cloud run.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,platform,re,shutil,statistics,subprocess,sys,time,zipfile
from check_outputs import load,relative_error,check
from edge_check import run_edges
from benchmark_tools import tune_kernel,compare_legacy,profile_kernel

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['naive','loops','tiled','fused','best','omp']

def capture(args):
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True)
    return dict(command=list(map(str,args)),returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())

def sysctl(key):
    r=capture(['/usr/sbin/sysctl','-n',key]);return r['stdout'] if r['returncode']==0 else None

def machine_data():
    keys=['machdep.cpu.brand_string','hw.model','hw.machine','hw.physicalcpu','hw.logicalcpu','hw.memsize','hw.cachelinesize','hw.l1dcachesize','hw.l1icachesize','hw.l2cachesize','hw.l3cachesize','hw.cpufrequency','hw.cpufrequency_max','hw.cpufrequency_min','hw.nperflevels']
    for level in [0,1]:
      for suffix in ['name','physicalcpu','logicalcpu','cpusperl2','l1dcachesize','l1icachesize','l2cachesize','cpufrequency','cpufrequency_max']:
        keys.append(f'hw.perflevel{level}.{suffix}')
    observations={k:sysctl(k) for k in keys}
    r=capture(['/usr/sbin/system_profiler','-json','SPHardwareDataType'])
    summary={}
    if r['returncode']==0:
        hardware=json.loads(r['stdout']).get('SPHardwareDataType',[{}])[0]
        safe_keys=['machine_name','machine_model','chip_type','cpu_type','current_processor_speed','number_processors','physical_memory']
        summary={k:hardware.get(k) for k in safe_keys if k in hardware}
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform=platform.platform(),architecture=platform.machine(),sysctl=observations,hardware=summary,os_version=capture(['/usr/bin/sw_vers']),affinity='OS-managed; no core pinning claimed',frequency_note='sysctl frequency values may be unavailable; no load-clock value is inferred',user_supplied_hardware={'model':'MacBookAir10,1','chip':'Apple M1','performance_cores':4,'efficiency_cores':4,'physical_cores':8,'logical_cores':8,'memory_gb':16})

def toolchain():
    brew=shutil.which('brew')
    if not brew:
        for p in ['/opt/homebrew/bin/brew','/usr/local/bin/brew']:
            if Path(p).is_file():brew=p;break
    if not brew:raise RuntimeError('Homebrew is required. Install it from https://brew.sh then run brew install llvm libomp python')
    llvm=capture([brew,'--prefix','llvm']);omp=capture([brew,'--prefix','libomp'])
    if llvm['returncode'] or omp['returncode']:raise RuntimeError('Run: brew install llvm libomp python')
    cc=Path(llvm['stdout'])/'bin/clang';lib=Path(omp['stdout'])/'lib';inc=Path(omp['stdout'])/'include'
    if not cc.is_file() or not (inc/'omp.h').is_file():raise RuntimeError('LLVM / OpenMP installation incomplete; run brew install llvm libomp python')
    return str(cc),['-I'+str(inc),'-L'+str(lib),'-Wl,-rpath,'+str(lib)]

def intensity(v,n,d):
    bytes_=16*n*n+16*n*d if v=='naive' else (16*n*n+24*n*d if v in ['loops','tiled'] else 24*n*d)
    return 4.0*n*n*d/bytes_

def execute(cmd,env,log):
    p=subprocess.run(list(map(str,cmd)),env=env,text=True,capture_output=True)
    log.write_text(log.read_text()+'$ '+' '.join(map(str,cmd))+'\n'+p.stdout+p.stderr+'\n' if log.exists() else '$ '+' '.join(map(str,cmd))+'\n'+p.stdout+p.stderr+'\n')
    if p.returncode:raise RuntimeError(f'Command failed ({p.returncode}): {cmd}\n{p.stdout}\n{p.stderr}')
    return p.stdout

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--tune',action='store_true',help='Measure ten cache/register tiles before choosing the best/omp configuration')
    parser.add_argument('--compare-legacy',action='store_true',help='Measure previous and current kernels in alternating order')
    parser.add_argument('--profile',action='store_true',help='Collect separate instrumented single-thread phase timings')
    args=parser.parse_args()
    if args.trials<1:raise RuntimeError('--trials must be positive')
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Run natively on your M1 Mac in an arm64 Terminal. This is not a Linux or Rosetta benchmark.')
    os.chdir(ROOT)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    evidence=ROOT/'output'/('mac_run_'+stamp);evidence.mkdir(parents=True)
    build=evidence/'build';build.mkdir();outputs=evidence/'outputs';outputs.mkdir()
    log=evidence/'run.log';log.write_text('Native Apple M1 CPU attention run\n')
    machine=machine_data();cc,omp_paths=toolchain();machine['compiler']=capture([cc,'--version'])
    (evidence/'machine.json').write_text(json.dumps(machine,indent=2))
    env=os.environ.copy();env.update(OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE',OMP_NUM_THREADS='1')
    env.pop('OMP_PLACES',None);env.pop('GOMP_CPU_AFFINITY',None)
    common=['-O3','-mcpu=apple-m1','-fopenmp-simd','-Wall','-Wextra','-Wno-unknown-pragmas']
    binaries={};commands=[]
    for v in VARIANTS:
        binary=build/v;binaries[v]=binary
        # Linking libomp for all drivers is harmless; only omp enables parallel
        # OpenMP regions. Others use simd-only compilation and request one thread.
        cmd=[cc,*common,*omp_paths,*(['-fopenmp'] if v=='omp' else []),ROOT/f'src/attn_{v}.c','-o',binary,'-lm','-lomp']
        commands.append(list(map(str,cmd)));execute(cmd,env,log)
    for name,source in [('bandwidth','bandwidth.c'),('compute_ceiling','compute_ceiling.c')]:
        cmd=[cc,'-O3','-mcpu=apple-m1','-fopenmp',*omp_paths,ROOT/'src'/source,'-o',build/name]
        execute(cmd,env,log);commands.append(list(map(str,cmd)))
    (evidence/'compile_commands.json').write_text(json.dumps(commands,indent=2))
    inputs=sorted((ROOT/'input').glob('*.bin'))
    if len(inputs)!=4:raise RuntimeError('Expected the four official input files')
    flags=[]
    if not args.tune:
        (ROOT/'output/kernel_config.mk').write_text('# Default untuned kernel for this run\nKERNEL_CFLAGS :=\n')
    if args.tune:
        flags=tune_kernel(cc,common,omp_paths,build,inputs,binaries,env,log,execute,commands,evidence)
        for v in ['best','omp']:
            cmd=[cc,*common,*omp_paths,*flags,*(['-fopenmp'] if v=='omp' else []),ROOT/f'src/attn_{v}.c','-o',binaries[v],'-lm','-lomp']
            commands.append(list(map(str,cmd)));execute(cmd,env,log)
    (evidence/'kernel_config.json').write_text(json.dumps(dict(best_flags=flags,fused_query_tile=48,fused_key_tile=128,fused_register_rows=4,tuned=args.tune),indent=2))
    physical=int(machine['sysctl'].get('hw.logicalcpu') or '8')
    thread_counts=sorted({t for t in [1,2,4,6,8,physical] if t<=physical})
    rows=[];raw=[]
    for v,t in [(v,1) for v in VARIANTS if v!='omp']+[('omp',t) for t in thread_counts]:
      for inp in inputs:
        env['OMP_NUM_THREADS']=str(t);samples=[]
        cmd=[binaries[v],inp,outputs/(v+'_'+inp.name)]
        for trial in range(args.trials):
            print(f'Running {v}, {inp.name}, {t} thread(s), trial {trial+1}/{args.trials}',flush=True)
            out=execute(cmd,env,log)
            m=re.search(r'N=\s*(\d+)\s+d=\s*(\d+)\s+threads=\s*(\d+)\s+time_ms=\s*([\d.]+)',out)
            if not m:raise RuntimeError('Unexpected driver output: '+out)
            n,d,actual=int(m[1]),int(m[2]),int(m[3]);ms=float(m[4])
            if actual!=t:raise RuntimeError(f'Driver reports {actual} threads, requested {t}')
            samples.append(ms)
        ms=statistics.median(samples);err=0.0
        if v!='naive':
            sh,a=load(outputs/('naive_'+inp.name));oh,b=load(cmd[-1]);err=relative_error(a,b)
            if sh!=oh or err>1e-4:raise RuntimeError(f'{v}: correctness failure {err}')
        rows.append(dict(variant=v,N=n,d=d,threads=t,time_ms=ms,gflops=4*n*n*d/(ms*1e6),arithmetic_intensity_flop_per_byte=intensity(v,n,d),make_check='REFERENCE' if v=='naive' else 'PASS',speedup_over_naive=0,parallel_speedup_vs_omp1='',parallel_efficiency='',relative_max_error=err))
        raw.append(dict(variant=v,N=n,d=d,threads=t,command=list(map(str,cmd)),trial_times_ms=samples,environment={k:env[k] for k in ['OMP_NUM_THREADS','OMP_DYNAMIC','OMP_PROC_BIND']},affinity='OS-managed'))
        (evidence/'raw_benchmarks.json').write_text(json.dumps(raw,indent=2))
    for r in rows:
        baseline=next(x for x in rows if (x['variant'],x['N'],x['d'])==('naive',r['N'],r['d']))
        r['speedup_over_naive']=baseline['time_ms']/r['time_ms']
        if r['variant']=='omp':
            one=next(x for x in rows if (x['variant'],x['N'],x['d'],x['threads'])==('omp',r['N'],r['d'],1))
            r['parallel_speedup_vs_omp1']=one['time_ms']/r['time_ms'];r['parallel_efficiency']=r['parallel_speedup_vs_omp1']/r['threads']
    with (evidence/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copyfile(evidence/'results.csv',ROOT/'results.csv')
    check(ROOT/'input',outputs)
    run_edges(binaries,env,evidence=evidence/'edge_checks.json')
    env['OMP_NUM_THREADS']='1'
    if args.compare_legacy:
        compare_legacy(cc,common,omp_paths,build,inputs,binaries,env,log,execute,commands,evidence,outputs,args.trials)
    if args.profile:
        profile_kernel(cc,common,omp_paths,build,inputs,flags,env,log,execute,commands,evidence,outputs)
    micro=[]
    for name in ['bandwidth','compute_ceiling']:
      for t in thread_counts:
        env['OMP_NUM_THREADS']=str(t);out=execute([build/name],env,log);print(out.strip(),flush=True)
        micro.append(dict(kind=name,threads=t,stdout=out))
    (evidence/'microbenchmarks.json').write_text(json.dumps(micro,indent=2))
    (evidence/'compile_commands.json').write_text(json.dumps(commands,indent=2))
    measured_python=['macos_run.py','benchmark_tools.py','edge_check.py','check_outputs.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*inputs,*sorted((ROOT/'src').glob('*.c')),*sorted((ROOT/'src').glob('*.h')),*[ROOT/'src'/name for name in measured_python]]}
    (evidence/'source_and_input_hashes.json').write_text(json.dumps(hashes,indent=2))
    machine['thread_counts']=thread_counts;machine['trials_per_configuration']=args.trials
    machine['kernel_config']=json.loads((evidence/'kernel_config.json').read_text())
    (evidence/'machine.json').write_text(json.dumps(machine,indent=2))
    bundle=ROOT/'Mac_Results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(evidence.rglob('*')):
            if p.is_file() and 'build' not in p.relative_to(evidence).parts and 'outputs' not in p.relative_to(evidence).parts:z.write(p,Path('mac_results')/p.relative_to(evidence))
    print('\nSUCCESS: native benchmarks and correctness checks finished.\nUpload this file: '+str(bundle),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:print('\nSTOPPED: '+str(e),file=sys.stderr);sys.exit(1)
