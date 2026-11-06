/* Hand-written attention kernels. No BLAS or external vector math library.
 * Compile on AVX-512/FMA x86 for the fast path; portable fallback otherwise.
 * The same core is used by best and omp, so scaling changes only scheduling.
 */
#ifndef BEST_KERNEL_H
#define BEST_KERNEL_H
#include <float.h>
#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_ZMM 1

/* exp(x) for x<=0: x=n*ln(2)+r; degree-7 Taylor on |r|<=ln(2)/2.
 * The first omitted term is < 5.2e-9 in absolute value on that interval.
 * Values below -87 contribute <1.65e-38 and are flushed to zero explicitly.
 */
static inline __m512 exp_nonpositive(__m512 x)
{
    __mmask16 tiny=_mm512_cmp_ps_mask(x,_mm512_set1_ps(-87.0f),_CMP_LT_OQ);
    x=_mm512_max_ps(x,_mm512_set1_ps(-87.0f));
    __m512 n=_mm512_roundscale_ps(_mm512_mul_ps(x,_mm512_set1_ps(1.4426950408889634f)),
                                _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512 r=_mm512_fnmadd_ps(n,_mm512_set1_ps(0.693145751953125f),x);
    r=_mm512_fnmadd_ps(n,_mm512_set1_ps(1.428606765330187e-6f),r);
    __m512 p=_mm512_set1_ps(1.0f/5040.0f);
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f/720.0f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f/120.0f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f/24.0f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f/6.0f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(0.5f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f));
    p=_mm512_fmadd_ps(p,r,_mm512_set1_ps(1.0f));
    return _mm512_mask_mov_ps(_mm512_scalef_ps(p,n),tiny,_mm512_setzero_ps());
}
#else
#define HAVE_ZMM 0
#endif

#include "neon_kernel.h"
static void normalize_row(float *row,int N)
{
#if HAVE_NEON
    neon_normalize(row,N);
    return;
#endif
    float mx=-FLT_MAX,sum=0;
    int j=0;
#if HAVE_ZMM
    __m512 m=_mm512_set1_ps(-FLT_MAX);
    for(;j+16<=N;j+=16) m=_mm512_max_ps(m,_mm512_loadu_ps(row+j));
    mx=_mm512_reduce_max_ps(m);
#endif
    for(;j<N;++j) if(row[j]>mx) mx=row[j];
    j=0;
#if HAVE_ZMM
    __m512 total=_mm512_setzero_ps(), vm=_mm512_set1_ps(mx);
    for(;j+16<=N;j+=16) {
        __m512 e=exp_nonpositive(_mm512_sub_ps(_mm512_loadu_ps(row+j),vm));
        _mm512_storeu_ps(row+j,e); total=_mm512_add_ps(total,e);
    }
    sum=_mm512_reduce_add_ps(total);
#endif
    for(;j<N;++j) {row[j]=expf(row[j]-mx);sum+=row[j];}
    const float inv=1.0f/sum;
    for(j=0;j<N;++j) row[j]*=inv;
}

/* A 4-query x 64-key microkernel: sixteen independent ZMM accumulators.
 * A 4x32 remainder handles inputs not divisible by 64.
 * T is a transposed key matrix, with adjacent key positions in SIMD lanes.
 */
static void scores(const float *Q,const float *T,float *S,int start,int rows,int N,int d)
{
    const float scale=1.0f/sqrtf((float)d);
    for(int r=0;r<rows;r+=4) {
        int nr=rows-r<4?rows-r:4;
        int j=0;
#if HAVE_ZMM
        if(nr==4) for(;j+64<=N;j+=64) {
            __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),b0=_mm512_setzero_ps(),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps(),b3=_mm512_setzero_ps(),c0=_mm512_setzero_ps(),c1=_mm512_setzero_ps(),c2=_mm512_setzero_ps(),c3=_mm512_setzero_ps(),e0=_mm512_setzero_ps(),e1=_mm512_setzero_ps(),e2=_mm512_setzero_ps(),e3=_mm512_setzero_ps();
            const float *q=Q+(size_t)(start+r)*d;
            for(int k=0;k<d;++k) {
                __m512 t0=_mm512_loadu_ps(T+(size_t)k*N+j+0);
                __m512 t1=_mm512_loadu_ps(T+(size_t)k*N+j+16);
                __m512 t2=_mm512_loadu_ps(T+(size_t)k*N+j+32);
                __m512 t3=_mm512_loadu_ps(T+(size_t)k*N+j+48);
                __m512 q0=_mm512_set1_ps(q[0*d+k]);
                a0=_mm512_fmadd_ps(q0,t0,a0);
                a1=_mm512_fmadd_ps(q0,t1,a1);
                a2=_mm512_fmadd_ps(q0,t2,a2);
                a3=_mm512_fmadd_ps(q0,t3,a3);
                __m512 q1=_mm512_set1_ps(q[1*d+k]);
                b0=_mm512_fmadd_ps(q1,t0,b0);
                b1=_mm512_fmadd_ps(q1,t1,b1);
                b2=_mm512_fmadd_ps(q1,t2,b2);
                b3=_mm512_fmadd_ps(q1,t3,b3);
                __m512 q2=_mm512_set1_ps(q[2*d+k]);
                c0=_mm512_fmadd_ps(q2,t0,c0);
                c1=_mm512_fmadd_ps(q2,t1,c1);
                c2=_mm512_fmadd_ps(q2,t2,c2);
                c3=_mm512_fmadd_ps(q2,t3,c3);
                __m512 q3=_mm512_set1_ps(q[3*d+k]);
                e0=_mm512_fmadd_ps(q3,t0,e0);
                e1=_mm512_fmadd_ps(q3,t1,e1);
                e2=_mm512_fmadd_ps(q3,t2,e2);
                e3=_mm512_fmadd_ps(q3,t3,e3);
            }
            __m512 sc=_mm512_set1_ps(scale);
            _mm512_storeu_ps(S+(size_t)(r+0)*N+j+0,_mm512_mul_ps(a0,sc));
            _mm512_storeu_ps(S+(size_t)(r+0)*N+j+16,_mm512_mul_ps(a1,sc));
            _mm512_storeu_ps(S+(size_t)(r+0)*N+j+32,_mm512_mul_ps(a2,sc));
            _mm512_storeu_ps(S+(size_t)(r+0)*N+j+48,_mm512_mul_ps(a3,sc));
            _mm512_storeu_ps(S+(size_t)(r+1)*N+j+0,_mm512_mul_ps(b0,sc));
            _mm512_storeu_ps(S+(size_t)(r+1)*N+j+16,_mm512_mul_ps(b1,sc));
            _mm512_storeu_ps(S+(size_t)(r+1)*N+j+32,_mm512_mul_ps(b2,sc));
            _mm512_storeu_ps(S+(size_t)(r+1)*N+j+48,_mm512_mul_ps(b3,sc));
            _mm512_storeu_ps(S+(size_t)(r+2)*N+j+0,_mm512_mul_ps(c0,sc));
            _mm512_storeu_ps(S+(size_t)(r+2)*N+j+16,_mm512_mul_ps(c1,sc));
            _mm512_storeu_ps(S+(size_t)(r+2)*N+j+32,_mm512_mul_ps(c2,sc));
            _mm512_storeu_ps(S+(size_t)(r+2)*N+j+48,_mm512_mul_ps(c3,sc));
            _mm512_storeu_ps(S+(size_t)(r+3)*N+j+0,_mm512_mul_ps(e0,sc));
            _mm512_storeu_ps(S+(size_t)(r+3)*N+j+16,_mm512_mul_ps(e1,sc));
            _mm512_storeu_ps(S+(size_t)(r+3)*N+j+32,_mm512_mul_ps(e2,sc));
            _mm512_storeu_ps(S+(size_t)(r+3)*N+j+48,_mm512_mul_ps(e3,sc));
        }
        if(nr==4) for(;j+32<=N;j+=32) {
            __m512 a0=_mm512_setzero_ps(),a1=a0,b0=a0,b1=a0,c0=a0,c1=a0,e0=a0,e1=a0;
            const float *q=Q+(size_t)(start+r)*d;
            for(int k=0;k<d;++k) {
                __m512 t0=_mm512_loadu_ps(T+(size_t)k*N+j);
                __m512 t1=_mm512_loadu_ps(T+(size_t)k*N+j+16);
                __m512 v=_mm512_set1_ps(q[k]);
                a0=_mm512_fmadd_ps(v,t0,a0);a1=_mm512_fmadd_ps(v,t1,a1);
                v=_mm512_set1_ps(q[d+k]);
                b0=_mm512_fmadd_ps(v,t0,b0);b1=_mm512_fmadd_ps(v,t1,b1);
                v=_mm512_set1_ps(q[2*d+k]);
                c0=_mm512_fmadd_ps(v,t0,c0);c1=_mm512_fmadd_ps(v,t1,c1);
                v=_mm512_set1_ps(q[3*d+k]);
                e0=_mm512_fmadd_ps(v,t0,e0);e1=_mm512_fmadd_ps(v,t1,e1);
            }
            __m512 sc=_mm512_set1_ps(scale);
            float *p=S+(size_t)r*N+j;
            _mm512_storeu_ps(p,_mm512_mul_ps(a0,sc));_mm512_storeu_ps(p+16,_mm512_mul_ps(a1,sc));
            _mm512_storeu_ps(p+N,_mm512_mul_ps(b0,sc));_mm512_storeu_ps(p+N+16,_mm512_mul_ps(b1,sc));
            _mm512_storeu_ps(p+2*N,_mm512_mul_ps(c0,sc));_mm512_storeu_ps(p+2*N+16,_mm512_mul_ps(c1,sc));
            _mm512_storeu_ps(p+3*N,_mm512_mul_ps(e0,sc));_mm512_storeu_ps(p+3*N+16,_mm512_mul_ps(e1,sc));
        }
#endif
#if HAVE_NEON
        if(nr==4) j=neon_scores4(Q,T,S,start,r,j,N,d,scale);
#endif
        for(;j<N;++j) for(int rr=0;rr<nr;++rr) {
            float a=0;
            for(int k=0;k<d;++k) a+=Q[(size_t)(start+r+rr)*d+k]*T[(size_t)k*N+j];
            S[(size_t)(r+rr)*N+j]=a*scale;
        }
    }
}

static void values(const float *S,const float *V,float *O,int start,int rows,int N,int d)
{
    for(int r=0;r<rows;r+=4) {
        int nr=rows-r<4?rows-r:4,k=0;
#if HAVE_ZMM
        if(nr==4) for(;k+64<=d;k+=64) {
            __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),b0=_mm512_setzero_ps(),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps(),b3=_mm512_setzero_ps(),c0=_mm512_setzero_ps(),c1=_mm512_setzero_ps(),c2=_mm512_setzero_ps(),c3=_mm512_setzero_ps(),e0=_mm512_setzero_ps(),e1=_mm512_setzero_ps(),e2=_mm512_setzero_ps(),e3=_mm512_setzero_ps();
            for(int j=0;j<N;++j) {
                __m512 v0=_mm512_loadu_ps(V+(size_t)j*d+k+0);
                __m512 v1=_mm512_loadu_ps(V+(size_t)j*d+k+16);
                __m512 v2=_mm512_loadu_ps(V+(size_t)j*d+k+32);
                __m512 v3=_mm512_loadu_ps(V+(size_t)j*d+k+48);
                __m512 p0=_mm512_set1_ps(S[(size_t)(r+0)*N+j]);
                a0=_mm512_fmadd_ps(p0,v0,a0);
                a1=_mm512_fmadd_ps(p0,v1,a1);
                a2=_mm512_fmadd_ps(p0,v2,a2);
                a3=_mm512_fmadd_ps(p0,v3,a3);
                __m512 p1=_mm512_set1_ps(S[(size_t)(r+1)*N+j]);
                b0=_mm512_fmadd_ps(p1,v0,b0);
                b1=_mm512_fmadd_ps(p1,v1,b1);
                b2=_mm512_fmadd_ps(p1,v2,b2);
                b3=_mm512_fmadd_ps(p1,v3,b3);
                __m512 p2=_mm512_set1_ps(S[(size_t)(r+2)*N+j]);
                c0=_mm512_fmadd_ps(p2,v0,c0);
                c1=_mm512_fmadd_ps(p2,v1,c1);
                c2=_mm512_fmadd_ps(p2,v2,c2);
                c3=_mm512_fmadd_ps(p2,v3,c3);
                __m512 p3=_mm512_set1_ps(S[(size_t)(r+3)*N+j]);
                e0=_mm512_fmadd_ps(p3,v0,e0);
                e1=_mm512_fmadd_ps(p3,v1,e1);
                e2=_mm512_fmadd_ps(p3,v2,e2);
                e3=_mm512_fmadd_ps(p3,v3,e3);
            }
            _mm512_storeu_ps(O+(size_t)(start+r+0)*d+k+0,a0);
            _mm512_storeu_ps(O+(size_t)(start+r+0)*d+k+16,a1);
            _mm512_storeu_ps(O+(size_t)(start+r+0)*d+k+32,a2);
            _mm512_storeu_ps(O+(size_t)(start+r+0)*d+k+48,a3);
            _mm512_storeu_ps(O+(size_t)(start+r+1)*d+k+0,b0);
            _mm512_storeu_ps(O+(size_t)(start+r+1)*d+k+16,b1);
            _mm512_storeu_ps(O+(size_t)(start+r+1)*d+k+32,b2);
            _mm512_storeu_ps(O+(size_t)(start+r+1)*d+k+48,b3);
            _mm512_storeu_ps(O+(size_t)(start+r+2)*d+k+0,c0);
            _mm512_storeu_ps(O+(size_t)(start+r+2)*d+k+16,c1);
            _mm512_storeu_ps(O+(size_t)(start+r+2)*d+k+32,c2);
            _mm512_storeu_ps(O+(size_t)(start+r+2)*d+k+48,c3);
            _mm512_storeu_ps(O+(size_t)(start+r+3)*d+k+0,e0);
            _mm512_storeu_ps(O+(size_t)(start+r+3)*d+k+16,e1);
            _mm512_storeu_ps(O+(size_t)(start+r+3)*d+k+32,e2);
            _mm512_storeu_ps(O+(size_t)(start+r+3)*d+k+48,e3);
        }
        if(nr==4) for(;k+32<=d;k+=32) {
            __m512 a0=_mm512_setzero_ps(),a1=a0,b0=a0,b1=a0,c0=a0,c1=a0,e0=a0,e1=a0;
            for(int j=0;j<N;++j) {
                __m512 v0=_mm512_loadu_ps(V+(size_t)j*d+k);
                __m512 v1=_mm512_loadu_ps(V+(size_t)j*d+k+16);
                __m512 p=_mm512_set1_ps(S[(size_t)r*N+j]);
                a0=_mm512_fmadd_ps(p,v0,a0);a1=_mm512_fmadd_ps(p,v1,a1);
                p=_mm512_set1_ps(S[(size_t)(r+1)*N+j]);
                b0=_mm512_fmadd_ps(p,v0,b0);b1=_mm512_fmadd_ps(p,v1,b1);
                p=_mm512_set1_ps(S[(size_t)(r+2)*N+j]);
                c0=_mm512_fmadd_ps(p,v0,c0);c1=_mm512_fmadd_ps(p,v1,c1);
                p=_mm512_set1_ps(S[(size_t)(r+3)*N+j]);
                e0=_mm512_fmadd_ps(p,v0,e0);e1=_mm512_fmadd_ps(p,v1,e1);
            }
            float *o=O+(size_t)(start+r)*d+k;
            _mm512_storeu_ps(o,a0);_mm512_storeu_ps(o+16,a1);
            _mm512_storeu_ps(o+d,b0);_mm512_storeu_ps(o+d+16,b1);
            _mm512_storeu_ps(o+2*d,c0);_mm512_storeu_ps(o+2*d+16,c1);
            _mm512_storeu_ps(o+3*d,e0);_mm512_storeu_ps(o+3*d+16,e1);
        }
#endif
#if HAVE_NEON
        if(nr==4) k=neon_values4(S,V,O,start,r,k,N,d);
#endif
        for(;k<d;++k) for(int rr=0;rr<nr;++rr) {
            float a=0;
            for(int j=0;j<N;++j) a+=S[(size_t)(r+rr)*N+j]*V[(size_t)j*d+k];
            O[(size_t)(start+r+rr)*d+k]=a;
        }
    }
}

#ifndef QUERY_BLOCK
#if HAVE_NEON
#define QUERY_BLOCK 16
#else
#define QUERY_BLOCK 64
#endif
#endif
static inline void materialized_attention(const float *Q,const float *K,const float *V,float *O,int N,int d,int parallel)
{
    float *T=malloc((size_t)d*N*sizeof(float));
    if(!T){fprintf(stderr,"allocation failed\n");exit(1);}
    /* Cache-blocked transpose, included in the timed attention() call. */
    #pragma omp parallel for collapse(2) if(parallel) schedule(static)
    for(int kk=0;kk<d;kk+=32)
      for(int jj=0;jj<N;jj+=32)
        for(int k=kk;k<d&&k<kk+32;++k)
          for(int j=jj;j<N&&j<jj+32;++j) T[(size_t)k*N+j]=K[(size_t)j*d+k];
    #pragma omp parallel if(parallel)
    {
        float *S=malloc((size_t)QUERY_BLOCK*N*sizeof(float));
        if(!S){fprintf(stderr,"allocation failed\n");exit(1);}
        #pragma omp for schedule(dynamic,1)
        for(int i=0;i<N;i+=QUERY_BLOCK) {
            int nr=N-i<QUERY_BLOCK?N-i:QUERY_BLOCK;
            scores(Q,T,S,i,nr,N,d);
            for(int r=0;r<nr;++r) normalize_row(S+(size_t)r*N,N);
            values(S,V,O,i,nr,N,d);
        }
        free(S);
    }
    free(T);
    (void)parallel;
}
#include "online_kernel.h"
static inline void best_attention(const float *Q,const float *K,const float *V,float *O,int N,int d,int parallel)
{
#ifdef ATTENTION_LEGACY
    materialized_attention(Q,K,V,O,N,d,parallel);
#else
    online_attention(Q,K,V,O,N,d,parallel);
#endif
}
#endif
