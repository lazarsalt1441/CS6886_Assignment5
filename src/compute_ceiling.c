/* Sustained FP32 FMA microbenchmark. This measures achieved throughput,
 * not architectural peak, and makes no claim of P/E core affinity on macOS. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
static float chains(uint64_t iterations) {
    float32x4_t a[16];
    for(int i=0;i<16;++i)a[i]=vdupq_n_f32(0.1f+i*0.001f);
    const float32x4_t mul=vdupq_n_f32(0.999999f),add=vdupq_n_f32(0.000001f);
    for(uint64_t j=0;j<iterations;++j) {
        /* Constant indices encourage Clang to keep all 16 chains in registers. */
        #define STEP(i) a[i]=vfmaq_f32(add,a[i],mul)
        STEP(0);STEP(1);STEP(2);STEP(3);STEP(4);STEP(5);STEP(6);STEP(7);
        STEP(8);STEP(9);STEP(10);STEP(11);STEP(12);STEP(13);STEP(14);STEP(15);
        #undef STEP
    }
    float sum=0;for(int i=0;i<16;++i)sum+=vaddvq_f32(a[i]);return sum;
}
#else
static float chains(uint64_t n){float a=0.1f;for(uint64_t i=0;i<n;++i)a=a*0.999999f+0.000001f;return a;}
#endif
int main(void) {
    const uint64_t iterations=8000000;
    double time[5],checksum=0;int actual_threads=0;
    for(int rep=0;rep<6;++rep) {
        double total=0,t0=omp_get_wtime();
        #pragma omp parallel reduction(+:total)
        {
            #pragma omp single
            actual_threads=omp_get_num_threads();
            total+=chains(iterations);
        }
        double elapsed=omp_get_wtime()-t0;
        if(rep)time[rep-1]=elapsed;checksum+=total;
    }
    for(int i=0;i<5;++i)for(int j=i+1;j<5;++j)if(time[j]<time[i]){double x=time[i];time[i]=time[j];time[j]=x;}
    printf("threads=%d median_s=%.9f achieved_fma_gflops=%.6f checksum=%.9f\n",actual_threads,time[2],(double)iterations*16*4*2*actual_threads/time[2]*1e-9,checksum);
    return 0;
}
