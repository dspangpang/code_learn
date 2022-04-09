#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<ctime>


__global__ void global_scan(float * d_out, float * d_in, int size){
    
    int kernelid = threadIdx.x + blockDim.x * blockIdx.x;
    // int threadid = threadIdx.x;
    int local_size = size;      //减少对全局变量的读取次数
    float out = 0.00f;
    d_out[kernelid] = d_in[kernelid];
    __syncthreads();

    for(int interval  = 1; interval < local_size; interval <<= 1 ){
        if(kernelid - interval >= 0){
            out = d_out[kernelid] + d_out[kernelid - interval];
        }
        __syncthreads();

        if(kernelid >= interval){
            d_out[kernelid] = out;
            out = 0.00f;
        }
    }
}

void scan_global(float * h_out, float * h_in, int size){

    const int MaxThreadsPerBlock = 1024;

    int threads, blocks;
    float *d_in, *d_out;

    int bytes = size * sizeof(float);

    threads = MaxThreadsPerBlock;
    blocks = (size + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;         //线程块上取整         


    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    global_scan<<<blocks, threads>>>(d_out, d_in, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);

}

void array_init(int size, float * array, float * sum){
    for(int i = 0; i < size; i++){
        array[i] = i ;
        if(i>0){
            sum[i] += array[i]+sum[i-1];
        }
        
    }
}

int main(){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cout << "ERROR" << std::endl;
    }
    const int ARRAY_SIZE = 1024*8;
    float h_in[ARRAY_SIZE] = {0};
    float h_out[ARRAY_SIZE] = {0};
 // float h_out1[ARRAY_SIZE] = {0};
    float sum[ARRAY_SIZE] = {0};

    clock_t start,end;

    array_init(ARRAY_SIZE, h_in, sum);
    
    start = clock();
    scan_global(h_out, h_in, ARRAY_SIZE);

    end = clock();
    std::cout << "global compute the answer is " << h_out[ARRAY_SIZE-1] << " run time is " << ((double) end -start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "cpu compute the answer is " << sum[ARRAY_SIZE-1] << std::endl;

}
