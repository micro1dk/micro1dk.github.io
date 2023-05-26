---
title: Element wise 연산
layout: default
parent: CUDA
nav_order: 3
---
# Element wise 연산

Element-wise 연산은 행렬의 각 요소에 동일한 연산을 적용하는 것이다. 각 요소는 독립적으로 다른 요소들과 상호작용하지 않고, 동일한 위치의 요소들끼리만 연산이 이루어진다.

행렬간 사칙연산의 경우 Operator를 작성하면 나중에 상당히 직관적으로 코딩할 수 있다.

```c
matA = matA + matB;
matA += matB
```



![](../../assets/images/cuda/element_wise.PNG)



커널 함수에 사용되는 파라미터에는 Device 메모리를 가리키는 포인터와 행렬의 정보를 넘겨준다.

커널을 실행하기 전 반드시 Device 메모리에 할당하는 과정이 필요하다. 만약 pitch로 할당했다면, 행렬의 Pitch 까지 파라미터로 넘겨주어야한다. 그러면 인덱싱은 `gy * pitch + gx`이 될것이다.

```c
__global__ void kernel_MatAddMat(
    float* Dst, const float* A, const float* B,
    int nrow, int ncol, bool minus
) {
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < nrow && gx < ncol) {
        int idx = gy * ncol + gx;
        if (!minus) {
            Dst[idx] = A[idx] + B[idx];
        } else {
            Dst[idx] = A[idx] - B[idx];
        }
    }
}

void exec_kernel_MatAddMat(
    float* Dst, const float* A, const float* B,
    int nrow, int ncol, bool minus, const int BLOCK_SIZE
) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(
        (ncol + dimBlock.x - 1) / dimBlock.x,
        (nrow + dimBlock.y - 1) / dimBlock.y,
        1
    );
    kernel_MatAddMat<<<dimGrid, dimBlock>>>(Dst, A, B, nrow, ncol, minus);
    cudaDeviceSynchronize();
}
```

두 행렬의 덧셈연산을 구현한 코드. 뺄셈, 곱셈, 나눗셈도 똑같은 형태로 커널을 만들면된다.

`cudaDeviceSynchronize()`는 GPU연산의 동기화를 위해 사용된다. 호출된 시점에서 GPU 연산이 완료될 때까지 대기한다. 쓰레드간 동기화는 `__syncthreads()`를 사용하는데 나중에 `shared_memory`를 사용할 때 활용하는 개념이다. 



다른 예로 Sigmoid 함수를 구현한 코드다.

```c
__device__ inline float sigmoid(float a) {
    return 1.0f / (1.0f + std::exp(-a));
}

__global__ void kernel_Sigmoid(
    float* Dst, const float* Src, 
    int nrow, int ncol
) {
    int gx = blockDim.x * blockIdx.x + threadIdx.x;
    int gy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gy < nrow && gx < ncol) {
        int idx = gy * ncol + gx;
        Dst[idx] = sigmoid(Src[idx]);
    }
}

void exec_kernel_Sigmoid(
    float* Dst, const float* Src, 
    int nrow, int ncol,
    const int BLOCK_SIZE
) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(
        (ncol + dimBlock.x - 1) / dimBlock.x,
        (nrow + dimBlock.y - 1) / dimBlock.y,
        1
    );
    kernel_Sigmoid<<<dimGrid, dimBlock>>>(
        Dst, Src, nrow, ncol
    );
    cudaDeviceSynchronize();
}
```

`__device__` 수식어는 함수를 GPU에서 실행하겠다는 뜻이다. 각 요소별로 Sigmoid 연산을 수행한다.