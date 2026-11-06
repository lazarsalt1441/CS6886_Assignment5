"""Native tile tuning and paired comparisons; original drivers remain unchanged."""
from pathlib import Path
import json, math, re, statistics, struct
from check_outputs import load, relative_error

ROOT = Path(__file__).resolve().parents[1]
TILES = [(24,64,4),(48,64,4),(48,128,4),(48,256,4),(96,128,4),
         (24,128,6),(48,64,6),(48,128,6),(48,256,6),(96,128,6)]

def kernel_flags(tile):
    return [f'-DATTENTION_QUERY_TILE={tile[0]}',f'-DATTENTION_KEY_TILE={tile[1]}',
            f'-DATTENTION_REGISTER_ROWS={tile[2]}']

def driver_time(text):
    found=re.search(r'time_ms=\s*([\d.]+)',text)
    if not found or float(found[1])<=0:raise RuntimeError('Invalid driver time: '+text)
    return float(found[1])

def validate(reference,output):
    shape,ref=load(reference);other,actual=load(output);error=relative_error(ref,actual)
    if shape!=other or error>1e-4:raise RuntimeError(f'Output check failed: {output}: {error}')
    return error

def compile_extra(cc,common,paths,source,binary,flags,env,log,execute,commands):
    cmd=[cc,*common,*paths,*flags,ROOT/'src'/source,'-o',binary,'-lm','-lomp']
    execute(cmd,env,log);commands.append(list(map(str,cmd)))

def tune_kernel(cc,common,paths,build,inputs,binaries,env,log,execute,commands,evidence):
    """Two reversed-order rounds; minimize geomean across all four inputs.

    Tiled is the inexpensive tuning reference. The complete run subsequently
    validates the selected kernel against naive and independent FP64 outputs.
    """
    refs={}
    for inp in inputs:
        out=build/('tune_reference_'+inp.name)
        execute([binaries['tiled'],inp,out],env,log);refs[inp.name]=out
    candidates=[]
    for tile in TILES:
        binary=build/('tune_%d_%d_%d'%tile);flags=kernel_flags(tile)
        compile_extra(cc,common,paths,'attn_best.c',binary,flags,env,log,execute,commands)
        candidates.append(dict(tile=tile,binary=binary,flags=flags,samples={}))
    for trial in range(2):
        for c in (candidates if trial==0 else candidates[::-1]):
            print(f'Tuning {c["tile"]}, round {trial+1}/2',flush=True)
            for inp in inputs:
                out=build/('tune_output_'+inp.name)
                text=execute([c['binary'],inp,out],env,log)
                validate(refs[inp.name],out)
                c['samples'].setdefault(inp.name,[]).append(driver_time(text))
    for c in candidates:
        c['geomean_ms']=math.exp(statistics.mean(math.log(statistics.median(x)) for x in c['samples'].values()))
    chosen=min(candidates,key=lambda c:c['geomean_ms'])
    data=dict(selected_tile=chosen['tile'],selected_flags=chosen['flags'],
              method='Minimum geometric mean of per-input median times; two reversed-order rounds',
              candidates=[{k:v for k,v in c.items() if k!='binary'} for c in candidates])
    (evidence/'kernel_tuning.json').write_text(json.dumps(data,indent=2))
    (ROOT/'output/kernel_config.mk').write_text('# Selected on this machine by macos_run.py --tune\nKERNEL_CFLAGS := '+' '.join(chosen['flags'])+'\n')
    print('Selected kernel tile: '+str(chosen['tile']),flush=True)
    return chosen['flags']

def compare_legacy(cc,common,paths,build,inputs,binaries,env,log,execute,commands,evidence,outputs,trials):
    names=['legacy_fused','fused','legacy_best','best'];programs=dict(binaries)
    for name,source in [('legacy_fused','attn_fused.c'),('legacy_best','attn_best.c')]:
        binary=build/name
        compile_extra(cc,common,paths,source,binary,['-DATTENTION_LEGACY'],env,log,execute,commands)
        programs[name]=binary
    records=[]
    for inp in inputs:
        samples={name:[] for name in names};errors={}
        for trial in range(trials):
            for name in (names if trial%2==0 else names[::-1]):
                print(f'Comparing {name}, {inp.name}, round {trial+1}/{trials}',flush=True)
                out=outputs/('comparison_'+name+'_'+inp.name)
                text=execute([programs[name],inp,out],env,log)
                errors[name]=validate(outputs/('naive_'+inp.name),out)
                samples[name].append(driver_time(text))
        for old,new in [('legacy_fused','fused'),('legacy_best','best')]:
            old_ms,new_ms=(statistics.median(samples[n]) for n in [old,new])
            records.append(dict(input=inp.name,previous_variant=old,new_variant=new,
                                previous_ms=old_ms,new_ms=new_ms,speedup=old_ms/new_ms,
                                previous_samples_ms=samples[old],new_samples_ms=samples[new],
                                relative_max_error=errors[new]))
    (evidence/'legacy_comparison.json').write_text(json.dumps(records,indent=2))
    return records

def profile_kernel(cc,common,paths,build,inputs,flags,env,log,execute,commands,evidence,outputs):
    records=[]
    for variant in ['fused','best']:
        binary=build/('profile_'+variant)
        compile_extra(cc,common,paths,'attn_'+variant+'.c',binary,
                      ['-DATTENTION_PROFILE',*(flags if variant=='best' else [])],env,log,execute,commands)
        for inp in inputs:
            out=outputs/('profile_'+variant+'_'+inp.name)
            text=execute([binary,inp,out],env,log)
            validate(outputs/('naive_'+inp.name),out)
            values=[{k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
                    for line in text.splitlines() if line.startswith('PROFILE ')]
            if len(values)!=6:raise RuntimeError('Expected warm-up plus five profiling measurements')
            middle=sorted(values[1:],key=lambda r:r['total_ms'])[2]
            n,d=struct.unpack('<ii',inp.read_bytes()[:8])
            records.append(dict(variant=variant,input=inp.name,representative=middle,
                                matmul_percent=100*(middle['scores_ms']+middle['values_ms'])/middle['total_ms'],
                                softmax_percent=100*middle['softmax_ms']/middle['total_ms'],
                                scores_gflops=2*n*n*d/(middle['scores_ms']*1e6),
                                values_gflops=2*n*n*d/(middle['values_ms']*1e6),raw=values,
                                note='Instrumented single-thread run includes timer overhead; not the results.csv timing.'))
    (evidence/'phase_profiles.json').write_text(json.dumps(records,indent=2))
    return records
