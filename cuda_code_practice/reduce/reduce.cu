#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<ctime>

__global__ void global_reduce(float * d_out, float * d_in){
    
    int kernelid = threadIdx.x + blockDim.x * blockIdx.x;
    int threadid = threadIdx.x;

    for(unsigned int s = blockDim.x / 2; s > 0 ; s >>= 1 ){
        if(threadid < s){
            d_in[kernelid] += d_in[kernelid + s];
        }
        __syncthreads();
    }
    if(threadid == 0){
        d_out[blockIdx.x] = d_in[kernelid];
    }
}

__global__ void shared_reduce(float * d_out, float * d_in){         //每一个线程快共享一个 shared memory
    
    extern __shared__ float sdata[];
    int threadid = threadIdx.x;
    int kernelid = threadIdx.x + blockDim.x * blockIdx.x;

    sdata[threadid] = d_in[kernelid];

    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0 ; s >>= 1 ){
        if(threadid < s){
            sdata[threadid] += sdata[threadid + s];
        }
        __syncthreads();
    }
    if(threadid == 0){
        d_out[blockIdx.x] = sdata[threadid];
    }
}

int find_block(int val){

    if(val & (val-1)){
        while (val & (val-1))
        {
            val &= (val-1); 
        }
        val <<= 1;
        return val;
    }
    else {
        return (val == 0)?(1):(val);
    }
        
}

void reduce_global(float * h_out, float * h_in, int size){

    const int MaxThreadsPerBlock = 1024;
    int threads, blocks;
    float *d_in, *d_out, *d_step;
    int bytes = size * sizeof(float);

    threads = MaxThreadsPerBlock;
    blocks = (size + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;         //线程块上取整 

    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_step, blocks * sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float));


    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    global_reduce<<<blocks, threads>>>(d_step, d_in);

    threads = find_block(blocks);
    blocks = 1;
    global_reduce<<<blocks, threads>>>(d_out, d_step);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);

}

void reduce_shared(float * h_out, float * h_in, int size){

    const int MaxThreadsPerBlock = 1024;
    int threads, blocks;
    float *d_in, *d_out, *d_step;
    int bytes = size * sizeof(float);

    threads = MaxThreadsPerBlock;
    blocks = (size + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;         //线程块上取整 

    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_step, blocks * sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float));


    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    shared_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_step, d_in);

    threads = find_block(blocks);
    blocks = 1;
    shared_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_step);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);

}
void array_init(int size, float * array, int * sum){
    for(int i = 0; i < size; i++){
        array[i] = i + 1 ;
        *sum += array[i];
    }
}
int main(int argc, char ** argv){
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cout << "ERROR" << std::endl;
    }
    const int ARRAY_SIZE = 1024*3;
    float h_in[ARRAY_SIZE] = {0};
    float h_out[1] = {0};
    float h_out1[1] = {0};
    int sum;

    clock_t start,end;
    
    array_init(ARRAY_SIZE, h_in, &sum);

    start = clock();
    reduce_global(h_out, h_in, ARRAY_SIZE);
    end = clock();

    std::cout << "global compute the answer is " << h_out[0] << " run time is " << ((double) end -start) / CLOCKS_PER_SEC << std::endl;

    start = clock();
    reduce_shared(h_out1, h_in, ARRAY_SIZE);
    end = clock();  
    std::cout << "shared compute the answer is " << h_out1[0] << " run time is " << ((double) end -start) / CLOCKS_PER_SEC <<std::endl;

    std::cout << "cpu compute the answer is " << sum << std::endl;

    return 0;
}