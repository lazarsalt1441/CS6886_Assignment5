/* Blocked online softmax attention. Included after the SIMD helpers in
 * best_kernel.h. Scratch scores are QUERY_TILE x KEY_TILE, independent of N.
 * No BLAS, external vector math, fast-math, or changes to starter drivers.
 */
#ifndef ONLINE_KERNEL_H
#define ONLINE_KERNEL_H
#ifndef ATTENTION_QUERY_TILE
#define ATTENTION_QUERY_TILE 48
#endif
#ifndef ATTENTION_KEY_TILE
#define ATTENTION_KEY_TILE 128
#endif
#ifndef ATTENTION_REGISTER_ROWS
#define ATTENTION_REGISTER_ROWS 4
#endif
#if ATTENTION_QUERY_TILE < 1 || ATTENTION_KEY_TILE < 1
#error Attention tile sizes must be positive
#endif
#if ATTENTION_REGISTER_ROWS != 4 && ATTENTION_REGISTER_ROWS != 6
#error Register rows must be 4 or 6
#endif

#if HAVE_ZMM && !defined(ATTENTION_SCALAR)
typedef __m512 ov;
#define OW 16
static inline ov oz(void){return _mm512_setzero_ps();}
static inline ov od(float x){return _mm512_set1_ps(x);}
static inline ov ol(const float*p){return _mm512_loadu_ps(p);}
static inline void os(float*p,ov x){_mm512_storeu_ps(p,x);}
static inline ov oa(ov a,ov b){return _mm512_add_ps(a,b);}
static inline ov om(ov a,ov b){return _mm512_mul_ps(a,b);}
static inline ov ob(ov a,ov b){return _mm512_sub_ps(a,b);}
static inline ov ox(ov a,ov b){return _mm512_max_ps(a,b);}
static inline ov of(ov a,ov b,ov c){return _mm512_fmadd_ps(a,b,c);}
static inline ov oe(ov a){return exp_nonpositive(a);}
static inline float ohsum(ov a){return _mm512_reduce_add_ps(a);}
static inline float ohmax(ov a){return _mm512_reduce_max_ps(a);}
#elif HAVE_NEON && !defined(ATTENTION_SCALAR)
typedef float32x4_t ov;
#define OW 4
static inline ov oz(void){return vdupq_n_f32(0);}
static inline ov od(float x){return vdupq_n_f32(x);}
static inline ov ol(const float*p){return vld1q_f32(p);}
static inline void os(float*p,ov x){vst1q_f32(p,x);}
static inline ov oa(ov a,ov b){return vaddq_f32(a,b);}
static inline ov om(ov a,ov b){return vmulq_f32(a,b);}
static inline ov ob(ov a,ov b){return vsubq_f32(a,b);}
static inline ov ox(ov a,ov b){return vmaxq_f32(a,b);}
static inline ov of(ov a,ov b,ov c){return vfmaq_f32(c,a,b);}
static inline ov oe(ov a){return neon_exp_nonpositive(a);}
static inline float ohsum(ov a){return vaddvq_f32(a);}
static inline float ohmax(ov a){return vmaxvq_f32(a);}
#else
typedef float ov;
#define OW 1
static inline ov oz(void){return 0;}
static inline ov od(float x){return x;}
static inline ov ol(const float*p){return *p;}
static inline void os(float*p,ov x){*p=x;}
static inline ov oa(ov a,ov b){return a+b;}
static inline ov om(ov a,ov b){return a*b;}
static inline ov ob(ov a,ov b){return a-b;}
static inline ov ox(ov a,ov b){return fmaxf(a,b);}
static inline ov of(ov a,ov b,ov c){return a*b+c;}
static inline ov oe(ov a){return expf(a);}
static inline float ohsum(ov a){return a;}
static inline float ohmax(ov a){return a;}
#endif
#define KS ATTENTION_KEY_TILE
#define QS ATTENTION_QUERY_TILE
#define MR ATTENTION_REGISTER_ROWS
#define ACCUMULATORS \
    ov a0=oz(),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0; \
    ov c0=a0,c1=a0,c2=a0,c3=a0,e0=a0,e1=a0,e2=a0,e3=a0
#if MR == 6
#define EXTRA_ACCUMULATORS ov f0=a0,f1=a0,f2=a0,f3=a0,g0=a0,g1=a0,g2=a0,g3=a0
#else
#define EXTRA_ACCUMULATORS
#endif
#define UPDATE(name,q) do { \
    ov qv=od(q); \
    name##0=of(qv,v0,name##0);name##1=of(qv,v1,name##1); \
    name##2=of(qv,v2,name##2);name##3=of(qv,v3,name##3); \
} while(0)
#define STORE(name,p) do { \
    os(p,name##0);os((p)+OW,name##1); \
    os((p)+2*OW,name##2);os((p)+3*OW,name##3); \
} while(0)

#ifdef ATTENTION_PROFILE
static double online_clock(void)
{
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+1e-9*(double)t.tv_nsec;
}
#define PHASE_START double phase_start=online_clock()
#define PHASE_END(index) do {if(!parallel)phase[index]+=online_clock()-phase_start;} while(0)
#else
#define PHASE_START
#define PHASE_END(index)
#endif

static void online_scores(const float *Q,const float *T,float *S,int rows,int keys,int d)
{
    const ov scale=od(1.0f/sqrtf((float)d));
    for(int r=0;r<rows;r+=MR){
        int nr=rows-r<MR?rows-r:MR,j=0;
        const float *q=Q+(size_t)r*d;
        if(nr==MR)for(;j+4*OW<=keys;j+=4*OW){
            ACCUMULATORS;
            EXTRA_ACCUMULATORS;
            for(int k=0;k<d;++k){
                const float *t=T+(size_t)k*KS+j;
                ov v0=ol(t),v1=ol(t+OW),v2=ol(t+2*OW),v3=ol(t+3*OW);
                UPDATE(a,q[k]);UPDATE(b,q[d+k]);
                UPDATE(c,q[2*d+k]);UPDATE(e,q[3*d+k]);
#if MR == 6
                UPDATE(f,q[4*d+k]);UPDATE(g,q[5*d+k]);
#endif
            }
            a0=om(a0,scale);a1=om(a1,scale);a2=om(a2,scale);a3=om(a3,scale);
            b0=om(b0,scale);b1=om(b1,scale);b2=om(b2,scale);b3=om(b3,scale);
            c0=om(c0,scale);c1=om(c1,scale);c2=om(c2,scale);c3=om(c3,scale);
            e0=om(e0,scale);e1=om(e1,scale);e2=om(e2,scale);e3=om(e3,scale);
#if MR == 6
            f0=om(f0,scale);f1=om(f1,scale);f2=om(f2,scale);f3=om(f3,scale);
            g0=om(g0,scale);g1=om(g1,scale);g2=om(g2,scale);g3=om(g3,scale);
#endif
            STORE(a,S+(size_t)r*KS+j);STORE(b,S+(size_t)(r+1)*KS+j);
            STORE(c,S+(size_t)(r+2)*KS+j);STORE(e,S+(size_t)(r+3)*KS+j);
#if MR == 6
            STORE(f,S+(size_t)(r+4)*KS+j);STORE(g,S+(size_t)(r+5)*KS+j);
#endif
        }
        for(int rr=0;rr<nr;++rr){
            int c=j;
            for(;c+OW<=keys;c+=OW){
                ov acc=oz();
                for(int k=0;k<d;++k)acc=of(od(q[(size_t)rr*d+k]),ol(T+(size_t)k*KS+c),acc);
                os(S+(size_t)(r+rr)*KS+c,om(acc,scale));
            }
            for(;c<keys;++c){
                float acc=0;
                #pragma omp simd reduction(+:acc)
                for(int k=0;k<d;++k)acc+=q[(size_t)rr*d+k]*T[(size_t)k*KS+c];
                S[(size_t)(r+rr)*KS+c]=acc/sqrtf((float)d);
            }
        }
    }
}

/* m/l are the maximum and UNNORMALIZED denominator over all preceding tiles.
 * The PV kernel applies alpha to the old numerator before adding this tile.
 * A pair of 4-lane NEON exponentials handles eight scores per iteration on M1.
 */
static void online_softmax(float *S,float *m,float *l,float *alpha,int rows,int keys)
{
    for(int r=0;r<rows;++r){
        float *s=S+(size_t)r*KS;int j=0;
        ov vmax=od(m[r]);
        for(;j+OW<=keys;j+=OW)vmax=ox(vmax,ol(s+j));
        float next=ohmax(vmax);
        for(;j<keys;++j)next=fmaxf(next,s[j]);
        alpha[r]=l[r]==0?0:(m[r]==next?1:expf(m[r]-next));
        ov vm=od(next),sum0=oz(),sum1=oz();j=0;
        for(;j+2*OW<=keys;j+=2*OW){
            ov p0=oe(ob(ol(s+j),vm)),p1=oe(ob(ol(s+j+OW),vm));
            os(s+j,p0);os(s+j+OW,p1);
            sum0=oa(sum0,p0);sum1=oa(sum1,p1);
        }
        for(;j+OW<=keys;j+=OW){ov p=oe(ob(ol(s+j),vm));os(s+j,p);sum0=oa(sum0,p);}
        float sum=ohsum(oa(sum0,sum1));
        for(;j<keys;++j){s[j]=expf(s[j]-next);sum+=s[j];}
        l[r]=l[r]*alpha[r]+sum;m[r]=next;
    }
}

#define INIT(name,p,x) \
    ov name##0=first?oz():om(ol(p),od(x)), \
       name##1=first?oz():om(ol((p)+OW),od(x)), \
       name##2=first?oz():om(ol((p)+2*OW),od(x)), \
       name##3=first?oz():om(ol((p)+3*OW),od(x))
static void online_values(const float *S,const float *V,float *O,const float *alpha,
                          int rows,int keys,int d,int first)
{
    for(int r=0;r<rows;r+=MR){
        int nr=rows-r<MR?rows-r:MR,k=0;
        if(nr==MR)for(;k+4*OW<=d;k+=4*OW){
            float *o=O+(size_t)r*d+k;
            INIT(a,o,alpha[r]);INIT(b,o+d,alpha[r+1]);
            INIT(c,o+2*d,alpha[r+2]);INIT(e,o+3*d,alpha[r+3]);
#if MR == 6
            INIT(f,o+4*d,alpha[r+4]);INIT(g,o+5*d,alpha[r+5]);
#endif
            for(int j=0;j<keys;++j){
                const float *v=V+(size_t)j*d+k;
                ov v0=ol(v),v1=ol(v+OW),v2=ol(v+2*OW),v3=ol(v+3*OW);
                UPDATE(a,S[(size_t)r*KS+j]);UPDATE(b,S[(size_t)(r+1)*KS+j]);
                UPDATE(c,S[(size_t)(r+2)*KS+j]);UPDATE(e,S[(size_t)(r+3)*KS+j]);
#if MR == 6
                UPDATE(f,S[(size_t)(r+4)*KS+j]);UPDATE(g,S[(size_t)(r+5)*KS+j]);
#endif
            }
            STORE(a,o);STORE(b,o+d);STORE(c,o+2*d);STORE(e,o+3*d);
#if MR == 6
            STORE(f,o+4*d);STORE(g,o+5*d);
#endif
        }
        for(int rr=0;rr<nr;++rr){
            float *o=O+(size_t)(r+rr)*d;const float *s=S+(size_t)(r+rr)*KS;
            int c=k;
            for(;c+OW<=d;c+=OW){
                ov acc=first?oz():om(ol(o+c),od(alpha[r+rr]));
                for(int j=0;j<keys;++j)acc=of(od(s[j]),ol(V+(size_t)j*d+c),acc);
                os(o+c,acc);
            }
            for(;c<d;++c){
                float acc=first?0:o[c]*alpha[r+rr];
                for(int j=0;j<keys;++j)acc+=s[j]*V[(size_t)j*d+c];
                o[c]=acc;
            }
        }
    }
}

static float *online_buffer(size_t count)
{
    void *p=NULL;
    /* M1 cache-line alignment also avoids split ZMM accesses on x86. */
    if(posix_memalign(&p,128,count*sizeof(float))!=0){
        fprintf(stderr,"allocation failed\n");exit(1);
    }
    return p;
}

static inline void online_attention(const float *Q,const float *K,const float *V,float *O,
                             int N,int d,int parallel)
{
#ifdef ATTENTION_PROFILE
    double phase[4]={0},total_start=online_clock();
#endif
    const int blocks=(N+KS-1)/KS;
    float *T=online_buffer((size_t)blocks*d*KS);
    #pragma omp parallel if(parallel)
    {
        float *S=online_buffer((size_t)QS*KS);
        /* One parallel region; its first implicit barrier publishes packed K.
         * Key-block-major packing keeps the reduction stride at KS, not N.
         */
        {PHASE_START;
        #pragma omp for schedule(static)
        for(int b=0;b<blocks;++b){
            int start=b*KS,keys=N-start<KS?N-start:KS;
            float *t=T+(size_t)b*d*KS;
            for(int k=0;k<d;++k)
                for(int j=0;j<keys;++j)t[(size_t)k*KS+j]=K[(size_t)(start+j)*d+k];
        }
        PHASE_END(0);}
        #pragma omp for schedule(dynamic,1)
        for(int i=0;i<N;i+=QS){
            int rows=N-i<QS?N-i:QS;
            float m[QS],l[QS],alpha[QS];
            for(int r=0;r<rows;++r){m[r]=-FLT_MAX;l[r]=0;}
            for(int b=0;b<blocks;++b){
                int start=b*KS,keys=N-start<KS?N-start:KS;
                {PHASE_START;online_scores(Q+(size_t)i*d,T+(size_t)b*d*KS,S,rows,keys,d);PHASE_END(1);}
                {PHASE_START;online_softmax(S,m,l,alpha,rows,keys);PHASE_END(2);}
                {PHASE_START;online_values(S,V+(size_t)start*d,O+(size_t)i*d,alpha,rows,keys,d,b==0);PHASE_END(3);}
            }
            {PHASE_START;
            for(int r=0;r<rows;++r){
                float *o=O+(size_t)(i+r)*d;ov inv=od(1.0f/l[r]);int k=0;
                for(;k+OW<=d;k+=OW)os(o+k,om(ol(o+k),inv));
                for(;k<d;++k)o[k]/=l[r];
            }
            PHASE_END(2);}
        }
        free(S);
    }
    free(T);(void)parallel;
#ifdef ATTENTION_PROFILE
    if(!parallel)printf("PROFILE pack_ms=%.6f scores_ms=%.6f softmax_ms=%.6f values_ms=%.6f total_ms=%.6f\n",
                       phase[0]*1e3,phase[1]*1e3,phase[2]*1e3,phase[3]*1e3,(online_clock()-total_start)*1e3);
#endif
}
#undef INIT
#undef STORE
#undef UPDATE
#undef ACCUMULATORS
#undef EXTRA_ACCUMULATORS
#undef QS
#undef KS
#undef MR
#undef OW
#undef PHASE_START
#undef PHASE_END
#endif
