/* Apple Silicon / AArch64 FP32 microkernels. No Accelerate, BLAS, AMX or SME.
 * 4 query rows x 16 columns = 16 independent 128-bit accumulators.
 * SIMD lanes cover keys (QK) or features (PV), never a serial dot reduction.
 */
#ifndef NEON_KERNEL_H
#define NEON_KERNEL_H
#if defined(__aarch64__) || defined(__arm64__) || defined(NEON_SCALAR_TEST)
#define HAVE_NEON 1
#ifdef NEON_SCALAR_TEST
#include "neon_scalar_test.h"
#else
#include <arm_neon.h>
#endif

static inline float32x4_t neon_exp_nonpositive(float32x4_t x)
{
    uint32x4_t tiny=vcltq_f32(x,vdupq_n_f32(-87.0f));
    x=vmaxq_f32(x,vdupq_n_f32(-87.0f));
    int32x4_t n=vcvtnq_s32_f32(vmulq_n_f32(x,1.4426950408889634f));
    float32x4_t nf=vcvtq_f32_s32(n);
    float32x4_t r=vfmsq_f32(x,nf,vdupq_n_f32(0.693145751953125f));
    r=vfmsq_f32(r,nf,vdupq_n_f32(1.428606765330187e-6f));
    float32x4_t p=vdupq_n_f32(1.0f/5040.0f);
    p=vfmaq_f32(vdupq_n_f32(1.0f/720.0f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f/120.0f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f/24.0f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f/6.0f),p,r);
    p=vfmaq_f32(vdupq_n_f32(0.5f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f),p,r);
    int32x4_t exponent=vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23);
    float32x4_t result=vmulq_f32(p,vreinterpretq_f32_s32(exponent));
    return vbslq_f32(tiny,vdupq_n_f32(0),result);
}

static void neon_normalize(float *row,int N)
{
    int j=0;float32x4_t m=vdupq_n_f32(-FLT_MAX);
    for(;j+4<=N;j+=4) m=vmaxq_f32(m,vld1q_f32(row+j));
    float mx=vmaxvq_f32(m);
    for(;j<N;++j) if(row[j]>mx) mx=row[j];
    j=0;float32x4_t total=vdupq_n_f32(0),vm=vdupq_n_f32(mx);
    for(;j+4<=N;j+=4) {
        float32x4_t e=neon_exp_nonpositive(vsubq_f32(vld1q_f32(row+j),vm));
        vst1q_f32(row+j,e);total=vaddq_f32(total,e);
    }
    float sum=vaddvq_f32(total);
    for(;j<N;++j){row[j]=expf(row[j]-mx);sum+=row[j];}
    float32x4_t inv=vdupq_n_f32(1.0f/sum);j=0;
    for(;j+4<=N;j+=4)vst1q_f32(row+j,vmulq_f32(vld1q_f32(row+j),inv));
    for(;j<N;++j)row[j]*=1.0f/sum;
}

/* Explicit variables keep register tiling visible to Clang on M1. */
#define NEON_ACCUMULATORS \
    float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0; \
    float32x4_t b0=a0,b1=a0,b2=a0,b3=a0; \
    float32x4_t c0=a0,c1=a0,c2=a0,c3=a0; \
    float32x4_t e0=a0,e1=a0,e2=a0,e3=a0
#define NEON_UPDATE(name,q) \
    name##0=vfmaq_n_f32(name##0,t0,q); \
    name##1=vfmaq_n_f32(name##1,t1,q); \
    name##2=vfmaq_n_f32(name##2,t2,q); \
    name##3=vfmaq_n_f32(name##3,t3,q)

static int neon_scores4(const float *Q,const float *T,float *S,int start,int r,int j,int N,int d,float scale)
{
    for(;j+16<=N;j+=16) {
        NEON_ACCUMULATORS;
        const float *q=Q+(size_t)(start+r)*d;
        for(int k=0;k<d;++k) {
            const float *t=T+(size_t)k*N+j;
            float32x4_t t0=vld1q_f32(t),t1=vld1q_f32(t+4),t2=vld1q_f32(t+8),t3=vld1q_f32(t+12);
            NEON_UPDATE(a,q[k]);NEON_UPDATE(b,q[d+k]);
            NEON_UPDATE(c,q[2*d+k]);NEON_UPDATE(e,q[3*d+k]);
        }
        float *p=S+(size_t)r*N+j;
        #define STORE_S(name,offset) \
            vst1q_f32(p+offset,vmulq_n_f32(name##0,scale)); \
            vst1q_f32(p+offset+4,vmulq_n_f32(name##1,scale)); \
            vst1q_f32(p+offset+8,vmulq_n_f32(name##2,scale)); \
            vst1q_f32(p+offset+12,vmulq_n_f32(name##3,scale))
        STORE_S(a,0);STORE_S(b,N);STORE_S(c,2*N);STORE_S(e,3*N);
        #undef STORE_S
    }
    return j;
}

static int neon_values4(const float *S,const float *V,float *O,int start,int r,int k,int N,int d)
{
    for(;k+16<=d;k+=16) {
        NEON_ACCUMULATORS;
        for(int j=0;j<N;++j) {
            const float *v=V+(size_t)j*d+k;
            float32x4_t t0=vld1q_f32(v),t1=vld1q_f32(v+4),t2=vld1q_f32(v+8),t3=vld1q_f32(v+12);
            NEON_UPDATE(a,S[(size_t)r*N+j]);NEON_UPDATE(b,S[(size_t)(r+1)*N+j]);
            NEON_UPDATE(c,S[(size_t)(r+2)*N+j]);NEON_UPDATE(e,S[(size_t)(r+3)*N+j]);
        }
        float *o=O+(size_t)(start+r)*d+k;
        #define STORE_O(name,offset) \
            vst1q_f32(o+offset,name##0);vst1q_f32(o+offset+4,name##1); \
            vst1q_f32(o+offset+8,name##2);vst1q_f32(o+offset+12,name##3)
        STORE_O(a,0);STORE_O(b,d);STORE_O(c,2*d);STORE_O(e,3*d);
        #undef STORE_O
    }
    return k;
}
#undef NEON_UPDATE
#undef NEON_ACCUMULATORS
#else
#define HAVE_NEON 0
#endif
#endif
