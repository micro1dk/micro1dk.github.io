---
title: 시작
layout: default
parent: CUDA
nav_order: 1
---

# 용어 몇 가지

## Thread

CUDA에서 실행되는 가장 작은 실행단위. 많은 수의 스레드가 동시에 실행되어 병렬처리를 수행한다. GPU에서 스레드는 작업을 독립적으로 처리하며, 각 스레드간 동기화 및 데이터 교환이 가능하다.



## Block

GPU에서 스레드의 그룹의 단위를 나타낸다. 하나의 블록은 한 번에 실행되는 스레드의 수를 정의한다. 블록 내 스레드는 동일한 GPU코어에서 실행된다. 

CUDA 아키텍처에서는 한 블록 내에서 실행되는 스레드의 수를 제한한다. 각 하드웨어마다 사양이 틀리니 사양을 확인해야한다. 보통은 1024개. 사양확인하는 방법은 아래에 설명해두었다.

1, 2, 3 차원 블록을 사용하여 구성한다.

블록은 dim3 데이터 타입으로 선언되며, 순서대로 x, y, z축의 크기를 나타낸다

```c
dim3 dimBlock(32, 32, 1); // x축 32, y축 32, z축 1
```

예를들어 2차원 블록의 크기가 (32, 32, 1) 인경우 총 32 * 32 = 1024개의 스레드로 구성된다. 

1024개로 맞추는것이 좋다.



## Grid

그리드는 GPU에서 실행되는 블록(BLOCK)의 그룹을 나타낸다.

다차원 구조로 정의되며, 1/2/3 차원 그리드를 사용한다.

그리드도 dim3 데이터 타입으로 선언되며, 순서대로 x, y, z축의 크기를 나타낸다.

각 축은 블록의 개수를 나타내며, 그리드의 크기는 각 축의 블록 개수를 곱한 값이다.



```c
int col = 2000;
int row = 2000;
dim3 dimBlock(32, 32, 1);
dim3 dimGrid(
    (col + dimBlock.x - 1) / dimBlock.x,
    (row + dimBlock.y - 1) / dimBlock.y,
    1
);
```

그리드의 크기는 블록의 수를 나타내는 값이므로, 블록의 수를 계산해야한다. 다음과 같은 공식을 사용한다.

```
블록 수 = (n + 블록크기 - 1) / 블록크기
```

계산하면 Grid의 크기는 (63, 63, 1)이다.

다시말해 한 그리드에는 크기가 (32, 32, 1)인 블록이  63 x 63 x 1 배열만큼 들어있다는 뜻이다. 





## Kernel

CUDA 프로그램에서 실행되는 병렬 컴퓨팅 작업 단위이다.

일반적으로 CUDA 프로그래밍 커널은 C언어로 작성되며 `.cu, .cuh`확장자의 파일에 작성한다. 커널 전용함수는  `__gloabl__` 수식어를 사용한다.

```c
__global__ void kernel_Dot(params..) {}
```



커널함수를 정의 했으면 커널실행코드를 작성할 수 있다. 

 `<<<grid_size, block_size>>>` 구문을 사용하여 커널을 실행한다.

```c
dim3 blockSize(32, 32, 1); // 2D 블록 크기
dim3 dimGrid(8, 8, 1); // 2D 그리드 크기

myKernel<<<dimGrid, blockSize>>>(params..);
```





## CUDA 사양확인

설치경로에서 extras/demo_suite 폴더에 접속 후 deviceQuery라는 파일을 실행시키면 된다. 

내 CUDA정보를 알려준다.

```bash
cuda-12/extras/demo_suite$ ./deviceQuery
```

```
mycamp@JeongSangyoung:/usr/local/cuda-12/extras/demo_suite$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 2050"
  CUDA Driver Version / Runtime Version          12.0 / 12.0
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 4096 MBytes (4294508544 bytes)
  (16) Multiprocessors, (128) CUDA Cores/MP:     2048 CUDA Cores
  GPU Max Clock rate:                            1155 MHz (1.15 GHz)
  Memory Clock rate:                             6001 Mhz
  Memory Bus Width:                              64-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.0, CUDA Runtime Version = 12.0, NumDevs = 1, Device0 = NVIDIA GeForce RTX 2050
Result = PASS
```

여기서 알야야할 정보는

*  CUDA Driver Version / Runtime Version: 12
   *  CUDA 버전은 12.
*  Total amount of shared memory per block:  49152 bytes
   *  공유메모리 할당용량
*  Maximum number of threads per block: 1024
   *  블록당 최대 쓰레드 제한
*  Texture alignment:  512 bytes
   * 메모리를 2D로 할당했을 때 pitch를 잡는 최소단위



