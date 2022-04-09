#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<ctime>

__global__ void local_histogram(int * d_out, int * d_in, int size_thread, int interval){
    
    int threadid = threadIdx.x;
    int remainder_3 ;

    for(int i = 0; i<size_thread; i++){
        remainder_3 = d_in[size_thread*threadid + i] % interval ;
        d_out[threadid * interval + remainder_3] += 1;
    }

}

__global__ void local_histogram_reduce(int * d_out, int * d_in, int interval){
    
    extern __shared__ float sdata[];
    int threadid = threadIdx.x;
    int kernelid = threadIdx.x + blockDim.x * blockIdx.x;

    sdata[threadid] = d_in[kernelid];

    __syncthreads();

    for(unsigned int s = blockDim.x / 2 ; s >= interval ; s >>= 1 ){
        if(threadid < s){
            sdata[threadid] += sdata[threadid + s];
        }
        __syncthreads();
    }
    if(threadid < 3){
        d_out[threadid] = sdata[threadid];
    }
}



bool one_block_histogram_reduce(int * h_out, int * h_in, int size, int interval, int size_thread){

    const int MaxThreadsPerBlock = 1024;
    int threads, blocks;
    int *d_in, *d_out, *d_step1;
    int bytes = size * sizeof(int);

    if(size / size_thread > MaxThreadsPerBlock){
        std::cout << "number overflow" << std::endl;
        return false;
    }
    threads = size / size_thread;
    blocks = 1 ;

    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_step1, interval * threads *sizeof(int));
    cudaMalloc((void**)&d_out, interval * sizeof(int));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    local_histogram<<<blocks, threads>>>(d_step1, d_in, size_thread, interval);

    threads = size / size_thread * interval;

    local_histogram_reduce<<<blocks, threads, interval * threads *sizeof(int)>>>(d_out, d_step1, interval);

    cudaMemcpy(h_out, d_out, sizeof(int) * interval, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_step1);

    return true;

}

void array_init(int size, int * array, int * sum, int interval){
    
    int remainder_3;

    for(int i = 0; i < size; i++){
        array[i] = i ;
        remainder_3 = i % interval;   
        sum[remainder_3] += 1;
    }
}

int main(){
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cout << "ERROR" << std::endl;
    }
    
    const int ARRAY_SIZE = 1024;
    const int interval = 3;
    const int size_thread = 16;

    int h_in[ARRAY_SIZE] = {0};
    int h_out[3] = {0};
    int sum[3] = {0};

    array_init(ARRAY_SIZE, h_in, sum, interval);
    one_block_histogram_reduce(h_out, h_in, ARRAY_SIZE, interval, size_thread);

    std::cout << "GPU answer is\n" << "group 1 : "<< h_out[0] << std::endl 
              <<"group 2 : "<< h_out[1] << std::endl << "group 3 : "<< h_out[2] << std::endl;

    std::cout << "CPU answer is\n" << "group 1 : "<< sum[0] << std::endl 
              <<"group 2 : "<< sum[1] << std::endl << "group 3 : "<< sum[2] << std::endl;

    return 0;
}