#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void sum_reduction(float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;
    float sum = 0;
    for (int i=tid; i<N; i+=blockDim.x * gridDim.x)
        sum += in[i];
    sdata[lane] = sum;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (lane < s) sdata[lane] += sdata[lane+s];
        __syncthreads();
    }
    if (lane == 0) out[blockIdx.x] = sdata[0];
}

double run_cuda(int N, int threads_per_block) {
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    float *h = new float[N], *d_in, *d_out;
    float *h_out = new float[blocks];
    for (int i =0; i<N; ++i) h[i]=1.0f;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, blocks*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);

    auto st = std::chrono::high_resolution_clock::now();
    sum_reduction<<<blocks, threads_per_block, threads_per_block*sizeof(float)>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, blocks*sizeof(float), cudaMemcpyDeviceToHost);
    double total=0;
    for (int i=0; i<blocks; ++i) total += h_out[i];
    auto en = std::chrono::high_resolution_clock::now();

    cudaFree(d_in); cudaFree(d_out);
    delete[] h; delete[] h_out;
    return std::chrono::duration<double, std::milli>(en-st).count();
}

int main(int argc, char* argv[]) {
    if (argc < 4) { std::cerr<<"Usage: ./cuda num_runs N threads_per_block\n"; return 1; }
    int R = atoi(argv[1]), N = atoi(argv[2]), T = atoi(argv[3]);
    double tot = 0;
    for (int i=0; i<R; ++i) tot += run_cuda(N, T);
    std::cout<<"CUDA avg time ("<<T<<"): "<< (tot/R) <<" ms\n";
    return 0;
}
