"""Rebuild the measured M1 submission from a native macos_run result folder.
Usage: python3 src/finalize_mac.py [--data src/evidence/mac_run]
Requires matplotlib for charts and pdflatex for the PDF. C kernels stay unchanged.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,re,shutil,statistics,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P_PEAK=3.204*4*4*2   # cited reference operating point and measured instruction throughput
E_PEAK=2.064*2*4*2
ALL_PEAK=4*P_PEAK+4*E_PEAK
BW_REFERENCE=4266*16/1000  # explicitly labelled 128-bit LPDDR4X platform model
SHAPES=[(1024,64),(2048,64),(4096,64),(4096,128)]
VARS=['naive','loops','tiled','fused','best']
COLORS=['#667680','#c68b32','#2a8588','#9a72a6','#155f9a']

def peak_model(threads):
    return min(threads,4)*P_PEAK+max(0,min(threads-4,4))*E_PEAK

def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')

def table(headers,rows,align=None,size='small'):
    align=align or 'l'+'r'*(len(headers)-1)
    return '\\begin{center}\n\\'+size+'\n\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{center}\n'

def load_data(data):
    machine=json.loads((data/'machine.json').read_text())
    raw=json.loads((data/'raw_benchmarks.json').read_text())
    edges=json.loads((data/'edge_checks.json').read_text())
    micro=json.loads((data/'microbenchmarks.json').read_text())
    hashes=json.loads((data/'source_and_input_hashes.json').read_text())
    for name,digest in hashes.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Source/input differs from the native run: {name}')
    with (data/'results.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['N','d','threads']:r[k]=int(r[k])
        for k in ['time_ms','gflops','arithmetic_intensity_flop_per_byte','speedup_over_naive','relative_max_error']:r[k]=float(r[k])
        for k in ['parallel_speedup_vs_omp1','parallel_efficiency']:
            if r[k]!='':r[k]=float(r[k])
        sample=next(x for x in raw if (x['variant'],x['N'],x['d'],x['threads'])==(r['variant'],r['N'],r['d'],r['threads']))
        if len(sample['trial_times_ms'])!=3 or statistics.median(sample['trial_times_ms'])!=r['time_ms']:
            raise RuntimeError('CSV/raw timing mismatch')
        expected=4*r['N']**2*r['d']/(r['time_ms']*1e6)
        if not math.isclose(expected,r['gflops'],rel_tol=1e-12):raise RuntimeError('GFLOP/s mismatch')
        if r['make_check'] not in ['PASS','REFERENCE'] or r['relative_max_error']>1e-4:raise RuntimeError('Correctness check failed')
        r['peak_model_gflops']=peak_model(r['threads'])
        r['percent_of_peak_model']=100*r['gflops']/r['peak_model_gflops']
        r['peak_model_basis']='P-first reference upper bound; OS placement/frequencies unmeasured'
    if len(rows)!=40 or len(edges)!=72:raise RuntimeError('Incomplete expected run')
    for r in rows:
        n=next(x for x in rows if (x['variant'],x['N'],x['d'])==('naive',r['N'],r['d']))
        assert math.isclose(n['time_ms']/r['time_ms'],r['speedup_over_naive'])
    bandwidth=[];compute=[]
    for x in micro:
        if x['kind']=='bandwidth':bandwidth.append((x['threads'],float(re.search(r'bandwidth_gbs=([\d.]+)',x['stdout'])[1])))
        else:compute.append((x['threads'],float(re.search(r'achieved_fma_gflops=([\d.]+)',x['stdout'])[1])))
    return machine,rows,raw,edges,bandwidth,compute

def get(rows,v,n=4096,d=64,t=1):
    return next(r for r in rows if (r['variant'],r['N'],r['d'],r['threads'])==(v,n,d,t))

def plots(rows,raw,bw,compute):
    p=ROOT/'plots';p.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':180})
    for n,d in SHAPES:
        rr=[get(rows,v,n,d) for v in VARS]+[r for r in rows if (r['variant'],r['N'],r['d'])==('omp',n,d)]
        labels=[r['variant'] if r['variant']!='omp' else f'omp\n{r["threads"]}t' for r in rr]
        times=[r['time_ms'] for r in rr]
        samples=[next(x['trial_times_ms'] for x in raw if (x['variant'],x['N'],x['d'],x['threads'])==(r['variant'],n,d,r['threads'])) for r in rr]
        error=np.array([[t-min(s) for t,s in zip(times,samples)],[max(s)-t for t,s in zip(times,samples)]])
        fig,ax=plt.subplots(figsize=(10.5,4.1),layout='constrained')
        bars=ax.bar(range(len(rr)),times,color=COLORS+['#5b854c']*5,yerr=error,capsize=3,error_kw={'elinewidth':.8})
        ax.set_xticks(range(len(rr)),labels);ax.set_yscale('log');ax.set_ylim(min(times)*.5,max(times)*2)
        ax.set(xlabel='Variant and OpenMP thread count',ylabel='Execution time (ms, log scale)',title=f'Apple M1 attention | N={n}, d={d}')
        ax.grid(axis='y',which='major',alpha=.2);ax.set_axisbelow(True)
        for b,t in zip(bars,times):ax.text(b.get_x()+b.get_width()/2,t*1.14,f'{t:.2f}',ha='center',fontsize=8)
        fig.savefig(p/f'time_N{n}_d{d}.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.7,3.8),layout='constrained')
    for n,d in SHAPES:
        rr=[r for r in rows if (r['variant'],r['N'],r['d'])==('omp',n,d)]
        ax.plot([r['threads'] for r in rr],[r['parallel_speedup_vs_omp1'] for r in rr],marker='o',label=f'N={n}, d={d}')
    ax.plot([1,8],[1,8],ls='--',lw=1,color='#888888',label='Identical-core ideal (context only)')
    ax.set(xlabel='Threads',ylabel='Speedup over omp, 1 thread',title='Measured M1 OpenMP scaling',xticks=[1,2,4,6,8]);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.savefig(p/'openmp_scaling.png');plt.close(fig)
    b1=dict(bw)[1];b8=dict(bw)[8];x=np.logspace(-2,3.4,500)
    fig,ax=plt.subplots(figsize=(8.3,4.7),layout='constrained')
    ax.loglog(x,np.minimum(b1*x,P_PEAK),color='#315d88',lw=2,label=f'1 P-core reference: {P_PEAK:.1f} GFLOP/s; measured 1t BW {b1:.1f} GB/s')
    ax.loglog(x,np.minimum(b8*x,ALL_PEAK),color='#93703e',lw=2,label=f'4P+4E reference: {ALL_PEAK:.1f} GFLOP/s; measured 8t BW {b8:.1f} GB/s')
    ax.loglog(x,np.minimum(BW_REFERENCE*x,ALL_PEAK),color='#888888',ls=':',lw=1.2,label=f'Platform DRAM model: {BW_REFERENCE:.2f} GB/s (reference only)')
    rr=[get(rows,v) for v in VARS]+[get(rows,'omp',t=8)]
    for r,c in zip(rr,COLORS+['#4a7b35']):
        ai=r['arithmetic_intensity_flop_per_byte'];gf=r['gflops'];label=r['variant'] if r['variant']!='omp' else 'omp (8t)'
        ax.scatter(ai,gf,s=55,color=c,edgecolors='white',linewidth=.7,zorder=5)
        offset={'naive':(7,4),'loops':(7,-10),'tiled':(7,5),'fused':(7,-7),'best':(7,5),'omp':(7,5)}[r['variant']]
        ax.annotate(label,(ai,gf),xytext=offset,textcoords='offset points',color=c,fontsize=9)
    ax.set(xlim=(.01,2400),ylim=(.3,1000),xlabel='Arithmetic intensity (FLOP/B; analytical traffic model)',ylabel='Useful-work throughput (GFLOP/s)',title='M1 roofline | N=4096, d=64')
    ax.grid(which='major',alpha=.2);ax.legend(loc='upper left',fontsize=7.1)
    fig.savefig(p/'roofline.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.3),layout='constrained')
    axes[0].plot(*zip(*bw),marker='o',color='#2a8588');axes[0].set(ylabel='Useful bandwidth (GB/s)',xlabel='Threads',title='384 MiB triad')
    axes[1].plot(*zip(*compute),marker='o',color='#155f9a');axes[1].set(ylabel='Achieved FP32 GFLOP/s',xlabel='Threads',title='Independent FMA microbenchmark')
    for ax in axes:ax.set_xticks([1,2,4,6,8]);ax.grid(alpha=.2)
    fig.savefig(p/'microbenchmarks.png');plt.close(fig)

def machine_spec(machine,bw,compute):
    b1=dict(bw)[1];b8=dict(bw)[8];best=max(b for _,b in bw)
    return f'''# Q1. Native Apple M1 measurement, 6 November 2026.
# Measured sysctl values and external architecture references are distinguished.
CPU model                       : Apple M1; MacBookAir10,1 (MacBook Air)
Microarchitecture               : Firestorm P cores + Icestorm E cores
Physical cores                  : 8
Logical cores (hardware threads): 8; no SMT
Performance / efficiency cores  : 4 performance / 4 efficiency (native sysctl)
Base clock (GHz)                : N/A as a fixed base; DVFS; actual run frequency not collected
Max single-core clock (GHz)     : Reference P=3.204, E=2.064 from Asahi/Linux T8103 operating points [3]
All-core clock under load (GHz) : Not exposed by collected sysctl; not measured or inferred
SIMD instruction set and width  : AArch64 Advanced SIMD (NEON), 128 bit, 4 FP32 lanes
FMA units per core              : P: 4 x 128-bit-equivalent FMLA/cycle; E: 2 (research measurements [2])
Vector registers per core       : 32 architectural 128-bit vector registers
L1 data cache (per core)        : P=128 KiB, E=64 KiB (native perflevel sysctl)
L2 cache (size, shared by)      : P cluster=12 MiB / 4 P cores; E cluster=4 MiB / 4 E cores
L3 cache (size, shared by)      : N/A as a reported conventional CPU L3; system-level cache not identified by this probe
Cache line size (bytes)         : 128 (native hw.cachelinesize)
RAM size (GB)                   : 16 GiB = 17.179869184 decimal GB; unified memory
RAM type, speed (MT/s), channels: Platform reference LPDDR4X-4266; 128-bit aggregate bus model (2 x 64-bit equivalent); physical subchannels not probed [4]
Operating system                : macOS 15.1.1 (24B91), arm64, native execution
Compiler and version            : Homebrew Clang 23.1.5; -O3 -mcpu=apple-m1 -fopenmp-simd; -fopenmp for omp

# Derived theoretical/reference values (upper bounds, not measured run clocks)
Peak FP32 GFLOP/s, one core     : P=3.204 x 4 FMLA/cycle x 4 lanes x 2 = {P_PEAK:.6f}; E=2.064 x 2 x 4 x 2 = {E_PEAK:.6f}
Peak FP32 GFLOP/s, all cores    : 4 x {P_PEAK:.6f} + 4 x {E_PEAK:.6f} = {ALL_PEAK:.6f}
Memory bandwidth (GB/s)         : Reference 4266 MT/s x (128/8) B = {BW_REFERENCE:.6f}; measured best={best:.6f} at 2t
Measured BW, one/eight threads  : {b1:.6f} / {b8:.6f} GB/s; useful bytes only, 384 MiB cached-store triad
Ridge point, one core (FLOP/B)  : P/theoretical BW={P_PEAK/BW_REFERENCE:.6f}; P/measured 1t BW={P_PEAK/b1:.6f}
Ridge point, all cores (FLOP/B) : All/theoretical BW={ALL_PEAK/BW_REFERENCE:.6f}; All/measured 8t BW={ALL_PEAK/b8:.6f}
Achieved FMA rate, one/all cores: {dict(compute)[1]:.6f} / {dict(compute)[8]:.6f} GFLOP/s (separate workload, not a hard ceiling)

# References: see report.pdf and src/references.md. No sustained clock is fabricated.
'''

def measured_table(rows):
    return table(['Variant','$N$','$d$','Time (ms)','GFLOP/s',r'\% peak$^\ast$','vs naive'],[[r['variant'],r['N'],r['d'],f'{r["time_ms"]:.3f}',f'{r["gflops"]:.2f}',f'{r["percent_of_peak_model"]:.2f}',f'{r["speedup_over_naive"]:.2f}$\\times$'] for r in rows])

def report(machine,rows,raw,edges,bw,compute):
    best=get(rows,'best');omp=get(rows,'omp',t=8);naive=get(rows,'naive')
    b1=dict(bw)[1];b8=dict(bw)[8]
    head=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,array,xcolor,hyperref,fancyhdr,xurl}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black,pdftitle={Assignment 5: Optimising Attention on Apple M1}}
\setlength{\emergencystretch}{1.5em}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\renewcommand{\arraystretch}{1.12}
\pagestyle{fancy}\fancyhf{}\lhead{CS6886: Systems for Deep Learning}\rhead{Assignment 5}\cfoot{\thepage}\setlength{\headheight}{13pt}
\newcommand{\question}[1]{\vspace{1mm}{\large\bfseries #1}\par\vspace{1mm}}
\begin{document}
{\LARGE\bfseries Optimising attention on Apple M1}\par
{\small Native MacBook Air measurements, 6 November 2026 \hfill CS6886, IIT Madras}
'''
    s=head+f'''
At $N=4096,d=64$, naive attention takes {naive['time_ms']:.3f} ms. The best single-threaded
variant takes {best['time_ms']:.3f} ms ({best['speedup_over_naive']:.2f}$\\times$ faster), and 8 OpenMP threads take
{omp['time_ms']:.3f} ms ({omp['speedup_over_naive']:.2f}$\\times$ faster). All supplied-input checks and 72 independent
FP64 checks pass. The C kernels use no BLAS or external vector-math library.

\\question{{Q1. Machine and peak calculations}}
The machine is a MacBook Air (MacBookAir10,1), with an Apple M1 and 16 GiB unified memory.
Native \\texttt{{sysctl}} confirms four performance (P) cores and four efficiency (E) cores,
eight physical/logical cores and no SMT. All timings are native arm64 on macOS 15.1.1,
using Homebrew Clang 23.1.5 with \\texttt{{-O3 -mcpu=apple-m1 -fopenmp-simd}};
only the OpenMP variant enables parallel OpenMP regions with \\texttt{{-fopenmp}}.
'''
    s+=table(['Property','Value / evidence'],[
      ['Core types','Firestorm P + Icestorm E; 4 of each (native probe; Apple [1])'],
      ['SIMD / registers','128-bit NEON, 4 FP32 lanes; 32 architectural vector registers'],
      ['FMLA throughput','P: 4/cycle; E: 2/cycle (instruction microbenchmarks [2])'],
      ['Clock reference','P: 3.204 GHz; E: 2.064 GHz, platform operating points [3]'],
      ['Base / sustained clock','DVFS; collected sysctl does not expose run-time core clocks'],
      ['L1D / L1I per P core','128 KiB / 192 KiB (native perflevel0 sysctl)'],
      ['L1D / L1I per E core','64 KiB / 128 KiB (native perflevel1 sysctl)'],
      ['L2 clusters','12 MiB shared by 4 P cores; 4 MiB shared by 4 E cores'],
      ['L3 / cache line','Conventional L3 not reported; system cache unprobed / 128 B'],
      ['Memory','16 GiB unified; LPDDR4X-4266 platform reference [4]']],align='lp{111mm}',size='footnotesize')
    s+=f'''
An FMA counts as two operations. Using the platform maximum operating points, the reference peaks are
\\begin{{align*}}
P_P&=3.204\\times4\\times4\\times2={P_PEAK:.3f}\\;\\mathrm{{GFLOP/s}},\\\\
P_E&=2.064\\times2\\times4\\times2={E_PEAK:.3f}\\;\\mathrm{{GFLOP/s}},\\\\
P_{{all}}&=4P_P+4P_E={ALL_PEAK:.3f}\\;\\mathrm{{GFLOP/s}}.
\\end{{align*}}
These are conditional architectural upper bounds, \\emph{{not measured sustained clocks}}. The
P operating point appears as a turbo entry in the Linux/Asahi platform description; its Linux
availability does not establish which macOS frequency was used. A fixed base/all-core frequency
cannot be reported from this run. Percent-of-peak uses $P_P$ for serial variants; OpenMP uses
$P(t)=\\min(t,4)P_P+\\max(0,t-4)P_E$, a P-first reference model, not verified core placement.

For the platform memory model, use a 128-bit aggregate interface (two 64-bit-equivalent widths):
$B_{{ref}}=4266\\times16/1000={BW_REFERENCE:.3f}$ GB/s. Memory type/rate come from platform
identification [4]; the bus width is an explicit model assumption, not a probe of physical
LPDDR subchannels. A 384 MiB triad measures useful bandwidth as $B=12n/t$ (two reads plus
one FP32 write), excluding cached-store write allocation. The one-thread rate is {b1:.3f} GB/s;
the best is {max(v for _,v in bw):.3f} GB/s at two threads; eight threads give {b8:.3f} GB/s.
Using measured bandwidth, the reference ridge points are $P_P/B_1={P_PEAK/b1:.3f}$ and
$P_{{all}}/B_8={ALL_PEAK/b8:.3f}$ FLOP/B. With theoretical $B_{{ref}}$, they are
{P_PEAK/BW_REFERENCE:.3f} and {ALL_PEAK/BW_REFERENCE:.3f} FLOP/B.
\\newpage
\\question{{Q2. Naive baseline}}
'''
    s+=measured_table([get(rows,'naive',n,d) for n,d in SHAPES])
    n1=get(rows,'naive',2048)['time_ms']/get(rows,'naive',1024)['time_ms'];n2=naive['time_ms']/get(rows,'naive',2048)['time_ms'];dr=get(rows,'naive',4096,128)['time_ms']/naive['time_ms']
    s+=f'''
The common work count is $F=4N^2d$: $2N^2d$ for $QK^T$ and $2N^2d$ for the value product.
Reported throughput is $F/(10^9t)$. Softmax, division, key transposition and allocation remain
in the time but not the useful-work numerator, following the starter driver.

At $d=64$, doubling $N$ multiplies time by {n1:.2f} and {n2:.2f}, close to the quadratic
prediction of 4. At $N=4096$, doubling $d$ multiplies time by {dr:.2f}: the products double
in work, while softmax still handles $N^2$ scores. Different cache/stride behaviour also matters.

The baseline computes each score as a separate dot reduction. Strict FP32 summation leaves
a dependent addition chain. Its value product uses $i,k,j$ loops, so $V[jd+k]$ is read with
a 256 B stride at $d=64$ and 512 B at $d=128$. The same probability row is traversed for
every output feature. This wastes locality and exposes reductions rather than independent
contiguous SIMD updates. These are code-based explanations; no per-stage timings or hardware
miss counters were collected on the Mac, so exact shares are not claimed.

\\textbf{{Protocol.}} Every original starter driver is preserved. Each invocation performs one warm-up
and takes the median of five timed calls; the CSV is the median of three independently warmed
invocations per configuration. Only \\texttt{{attention()}} is timed; file I/O is excluded, while all
temporary allocations and packing are included. Configurations run sequentially. macOS owns
P/E placement; no CPU affinity or fixed core frequency is claimed. Three-trial ranges on the
time plots are variability indicators, not confidence intervals.
\\begin{{center}}\\includegraphics[width=0.98\\linewidth]{{plots/time_N4096_d64.png}}\\end{{center}}
The full CSV contains every input/variant/thread configuration. All four required per-input
time plots are supplied. One noisy invocation cannot change a row to its best-case time:
the report consistently uses the three-invocation median.
\\vfill
{{\\footnotesize $^\\ast$ Peak percentages refer to the Q1 reference architecture/frequency model.
GB and GFLOP use decimal units; KiB/MiB/GiB use binary units.}}
\\newpage
\\question{{Q3. Single-threaded variants}}
'''
    s+=measured_table([get(rows,v,n,d) for v in VARS[1:] for n,d in SHAPES])
    s+=r'''
\textbf{(a) Reordered loops.} Transpose keys to $T[k,j]$ and use $i,k,j$ score loops. Adjacent
key positions become independent SIMD lanes. For $SV$, use $i,j,k$ so each probability
updates contiguous output features from a contiguous row of $V$. This removes strided
column access, but repeatedly updates score/output accumulators and retains the full score matrix.

\textbf{(b) Cache/register tiling.} Use $32\times128$ outer query/key tiles around $4\times32$
local accumulator tiles. Four query rows reuse each key/value vector; complete reductions
are kept in accumulators before storing results. It improves the $4096\times64$ case, but is
slightly slower than loops at $4096\times128$ (422.245 vs 419.289 ms). Tiling therefore has
shape-dependent benefits; local-array/control overhead can offset cache reuse. The difference
is small and is not evidence of a fundamental regression without more trials.

\textbf{(c) Row fusion.} Compute one score row, stable softmax, and its output immediately.
At $N=4096$, score storage falls from 64 MiB to 16 KiB. This version is still slower than
tiled at $d=64$: per-score dot reductions and lost multi-query reuse offset the storage saving.
At $d=128$, loops/tiled/fused are all around 419--422 ms, rather than monotonically improving.

\textbf{(d) Best M1 implementation.} Fuse blocks of 16 query rows; each NEON microkernel handles
four query rows by 16 keys/features, using sixteen independent 128-bit accumulators. Reuse
each loaded key/value vector across four queries and keep partial sums in registers.
A custom vector exponential accelerates stable softmax. At $N=4096$, each block needs 256 KiB
scratch, alongside 1 MiB of transposed keys and 1 MiB of values ($d=64$), fitting the P-cluster
L2. Packing is still timed. Scalar tails handle odd dimensions; the AArch64 fast path was
actually exercised in the submitted native Mac run. This is the best implemented/tested
variant, without a claim of global optimality or a measured M1 block-size search.

The remaining limits include FMA/load/broadcast throughput, softmax, packing, cache traffic
and OS scheduling. Exact bottleneck shares would require profiling beyond this run.
\newpage
\question{Q4. OpenMP}
The same compute kernel is used in \texttt{best} and \texttt{omp}. Key transposition is parallelised
over two-dimensional tiles. Query blocks use \texttt{schedule(dynamic,1)}: workers own private
scratch and disjoint output rows, and read shared keys/values. Dynamic block assignment lets
faster workers take more work on the asymmetric 4P+4E CPU. There is no cross-thread output reduction.

The controller disables dynamic team resizing and core binding. Thread placement remains
OS-managed. Define $S_t=T_{omp,1}/T_{omp,t}$ and $E_t=S_t/t$, using the one-thread OpenMP
implementation as the scaling baseline rather than the separately compiled serial binary.
'''
    rr=[]
    for n,d in SHAPES:
      for r in [x for x in rows if (x['variant'],x['N'],x['d'])==('omp',n,d)]:
        rr.append([n,d,r['threads'],f'{r["time_ms"]:.3f}',f'{r["gflops"]:.2f}',f'{r["parallel_speedup_vs_omp1"]:.3f}',f'{100*r["parallel_efficiency"]:.1f}'])
    s+=table(['$N$','$d$','Threads','Time (ms)','GFLOP/s','$S_t$',r'$E_t$ (\%)'],rr,size='footnotesize')
    s+=f'''
For $N=4096,d=64$, eight threads yield {omp['parallel_speedup_vs_omp1']:.3f}$\\times$ scaling and
{100*omp['parallel_efficiency']:.1f}\\% efficiency, or {omp['speedup_over_naive']:.2f}$\\times$ over naive.
Eight identical full-speed cores would suggest an 8$\\times$ line, but this CPU has four slower
E cores, shared caches/memory and scheduling overhead. That line is context, not an achievable
P-core-only target. Six threads are slower than four for $2048\\times64$ and $4096\\times128$;
eight are fastest for all four supplied shapes. The measurements show non-monotonic scaling,
not which thread was on which core. Contention, dynamic scheduling, placement and thermal/DVFS
effects are plausible contributors, not separately established causes.
\\newpage
\\question{{Q5. Roofline}}
\\begin{{center}}\\includegraphics[width=0.98\\linewidth]{{plots/roofline.png}}\\end{{center}}
'''
    s+=r'''
\textbf{Analytical traffic model.} Start with $16Nd$ bytes for reading $Q,K,V$ and writing $O$.
For materialised scores, add $16N^2$ bytes across stages: score write, softmax read/write,
then value-product read. Softmax's internal row passes are assumed to hit cache. A transposed
key buffer adds a conservative $8Nd$ bytes for its write/read. Fused row/block scratch is
assumed cache-resident. Ignore write allocation and additional evictions. Hence these
idealised intensities are not measured DRAM traffic and may overestimate actual intensity.
'''
    ai=[]
    for v,formula in [('naive',r'$16N^2+16Nd$'),('loops',r'$16N^2+24Nd$'),('tiled',r'$16N^2+24Nd$'),('fused',r'$16Nd$'),('best',r'$24Nd$'),('omp',r'$24Nd$')]:
        r=omp if v=='omp' else get(rows,v);ai.append([v if v!='omp' else 'omp (8t)',formula,f'{r["arithmetic_intensity_flop_per_byte"]:.3f}',f'{r["gflops"]:.2f}'])
    s+=table(['Variant','Modelled bytes','$I=F/B$ (FLOP/B)','GFLOP/s'],ai,align='llrr',size='footnotesize')
    s+=r'''
The rooflines use $\min(BI,P)$ with Q1's reference peak and either measured sustained bandwidth
or the explicitly assumed platform memory model. Serial points belong against the one-P-core
curve; the OpenMP point uses the 4P+4E curve. All points are beyond the corresponding ridge.
Thus modelled DRAM bandwidth is not the tight ceiling; this does not prove the kernels are
purely compute-bound. Naive suffers reduction/stride/cache-level costs; loops exposes SIMD
but repeats accumulator traffic; tiled reuses accumulators. Fused improves storage without
equivalent multi-row arithmetic reuse. Best adds explicit SIMD/register tiling and faster
softmax. OpenMP raises throughput but does not saturate the aggregate reference peak.
\newpage
\question{Validation, auxiliary measurements and reproduction}
'''
    maxerr=max(r['relative_max_error'] for r in rows);edgeerr=max(e['relative_error_fp64'] for e in edges)
    s+=f'''
\\textbf{{Validation.}} All 36 non-reference input/configuration rows pass the required
$\\max|O-O_{{naive}}|/\\max|O_{{naive}}|\\le10^{{-4}}$, including all 20 OpenMP configurations.
The worst supplied-input error is {maxerr:.3g}. All 72 independent FP64 checks pass, with a
worst error of {edgeerr:.3g}; these include odd shapes, tails, zero/uniform logits and large logits.
SHA-256 matches all returned C/header sources and four original inputs to the packaged files.
The original naive source and all starter drivers are preserved. \\texttt{{make check}} rejects
missing outputs, invalid headers/lengths, non-finite values and excessive error.

\\textbf{{Vector softmax approximation.}} Subtract the row maximum, then reduce $x=n\\ln2+r$ with
$n=\\operatorname{{round}}(x\\log_2e)$ and $|r|\\le\\ln2/2$. Evaluate
$e^x\\approx2^n\\sum_{{k=0}}^7r^k/k!$ with NEON FMAs. Values below $-87$ are set to zero
(each contribution below $1.65\\times10^{{-38}}$); scalar tails use \\texttt{{expf}}. The first omitted
polynomial term is below $5.2\\times10^{{-9}}$, but final accuracy also includes FP32 range-reduction,
FMA and summation rounding. Output checks establish tolerance on the tested cases, not exactness.
\\begin{{center}}\\includegraphics[width=0.98\\linewidth]{{plots/microbenchmarks.png}}\\end{{center}}
'''
    s+=table(['Threads','Triad GB/s','Achieved FMA GFLOP/s'],[[t,f'{dict(bw)[t]:.3f}',f'{dict(compute)[t]:.3f}'] for t,_ in bw],size='footnotesize')
    s+=r'''
The FMA test runs sixteen independent NEON chains, one warm-up and five measurements, counting
$16\times4\times2$ FLOPs per iteration per thread. It reports achieved rates, not a hard roof:
one attention shape exceeds its one-thread rate because instruction mix, timing and operating
conditions differ. The triad's best bandwidth occurs at two threads; more workers add contention
and scheduling costs rather than increasing the measured useful byte rate. No cache-miss,
power or sustained-frequency counters were collected.

\textbf{Reproduce.} On the M1, install Homebrew LLVM/libomp/Python, then run
\texttt{bash src/run\_macos.sh}. It compiles the kernels, runs all configurations/checks and
collects a result ZIP. \texttt{make all; make check} gives the starter-style workflow.
\texttt{python3 src/finalize\_mac.py} rebuilds this PDF, CSV/specification and plots from the
included native evidence (requires Matplotlib, NumPy and pdflatex). Use \texttt{--data PATH}
to supply another complete native run. Regeneration validates source hashes and raw medians.

Native evidence is under \path{src/evidence/mac_run/}. Local root paths are shortened
for portability. The main CSV preserves the measured
columns and adds clearly labelled reference-peak percentages. No earlier cloud timings,
stage-profile figures or tuning measurements are mixed into the Mac report.
\newpage
\question{References and run trace}
'''
    # Separate compact provenance/reference page avoids crowding the numerical report.
    refs=[
      ('[1] Apple, MacBook Air (M1, 2020) technical specifications.','https://support.apple.com/en-us/111883'),
      ('[2a] Dougall Johnson, Apple Microarchitecture Research: Firestorm SIMD/FP, FMLA vector 4S (0.25 cycles per instruction).','https://dougallj.github.io/applecpu/firestorm-simd.html'),
      ('[2b] Dougall Johnson, Icestorm SIMD/FP, FMLA vector 4S (0.5 cycles per instruction). These are primary reverse-engineering measurements, not Apple specifications.','https://dougallj.github.io/applecpu/icestorm-simd.html'),
      ('[3] Asahi Linux contributors, Linux T8103 device tree: 3.204 GHz P turbo / 2.064 GHz E operating points; these are reference values, not observed macOS clocks.','https://github.com/torvalds/linux/blob/master/arch/arm64/boot/dts/apple/t8103.dtsi'),
      ('[4] iFixit, M1 MacBook Pro and Air Teardowns: primary chip identification of LPDDR4X-4266. The report explicitly assumes a 128-bit aggregate interface for the theoretical bandwidth model.','https://www.ifixit.com/News/46884/m1-macbook-teardowns-something-old-something-new'),
      ('[5] Apple, Tuning your code\'s performance for Apple silicon: asymmetric cores and scheduling.','https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon/'),
      ('[6] Assignment starter, commit 29f5b9bf4aa89f74f425b1797cc46cdb457f5569.','https://github.com/NMJ0/Assignment-5-Systems-for-Deep-learning')]
    for title,url in refs:s+=esc(title)+'\n\n\\url{'+url+'}\n\n'
    s+=r'''
\question{What is measured, assumed or unavailable}
'''
    s+=table(['Category','Treatment'],[
      ['Measured','Native times, errors, cache sizes, RAM capacity, cores, compiler, triad/FMA rates'],
      ['Reference model','P/E instruction throughput and maximum operating points; LPDDR4X interface model'],
      ['Unavailable','Actual per-core placement and clocks under load, cache-miss counters, package power'],
      ['Not inferred','Eight P cores; a guaranteed base/all-core frequency; per-stage timing shares'],
      ['Not established','Global optimality, definitive causes of every scaling fluctuation']],align='lp{111mm}',size='footnotesize')
    s+=r'''
The full package retains the exact native C sources rather than substituting a subsequently
modified kernel. To reproduce the submitted results, use the same compiler/flags and report
all new timings when conditions change. The native runner prints its progress and fails if
compilation or a numerical check fails. The original binary inputs remain under \texttt{input/}.
The public GitHub URL is a submission step; it is not a performance result and must be supplied
when the repository is published.
\end{document}
'''
    (ROOT/'src/report.tex').write_text(s)
    out=ROOT/'build/report';out.mkdir(parents=True,exist_ok=True)
    for _ in range(2):
        p=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),str(ROOT/'src/report.tex')],cwd=ROOT,text=True,capture_output=True)
        (out/'compile_stdout.txt').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout[-6000:])
    shutil.copyfile(out/'report.pdf',ROOT/'report.pdf')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='src/evidence/mac_run');parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);data=Path(args.data).resolve()
    machine,rows,raw,edges,bw,compute=load_data(data)
    with (ROOT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'machine_spec.txt').write_text(machine_spec(machine,bw,compute))
    plots(rows,raw,bw,compute)
    if not args.plots_only:report(machine,rows,raw,edges,bw,compute)
    print('Validated native data; generated CSV, machine specification, plots'+(' and report.' if not args.plots_only else '.'))

if __name__=='__main__':main()
