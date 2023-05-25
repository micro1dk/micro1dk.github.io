---
title: Matrix class
layout: default
parent: 신경망
nav_order: 2
---

# Matrix class

Matrix 클래스를 구성할 때 고려해야할 요소들이 있다.

* 데이터 저장방식
  * CUDA에는 메모리 저장방식이 몇 가지 있다. 1D, 2D_pitch, 3D_pitch등
  * 본인은 1D를 활용하여 행과 열 인덱스를 활용하여 원소에 접근하는 방식으로 개발할 것이다. 이론상 2D pitch할당방식으로 하는것이 더 빠르지만, 여러 테스트를 해본 결과 차이가 없었다.
* 연산
  * 행렬과 행렬간 연산, 행렬과 벡터간 연산, 행렬과 상수간 연산
  * Operator를 활용하면 신경망 코딩을 직관적으로 작성할 수 있다.
* 클래스 초기화, 클래스 복사, 클래스 소멸자
* 메모리 관리
  * shared_ptr를 사용하면 관리가 쉽다고는 하지만, Matrix 인스턴스를 계속 생성해야하기 때문에 오버헤드가 클 것같아 사용하지는 않을것임.



# Host 메모리, Device 메모리

Host 메모리와 Device 메모리는 CUDA 프로그래밍에서 사용되는 단어다.



### Host 메모리

CPU가 직접 엑세스할 수 있는 메모리 영역. C 프로그래밍할 때 일반적으로 할당하는 메모리영역. 

메모리를 할당하는 데 비용이 크다. 따라서 Matrix 인스턴스를 생성했을 때는 Host메모리를 할당하지 않는다.



### Device 메모리

GPU가 직접 엑세스할 수 있는 메모리 영역. GPU는 CUDA커널에서 실행되며 병렬 처리를 수행하기 위해 Device메모리를 사용함

병렬 연산을 위해 Matrix 인스턴스 생성시 Device메모리를 할당해야한다.



## 생성자, 소멸자, 복사생성자

Matrix의 생성자에는 행렬의 크기를 인자로 받고 그 크기만큼 GPU 메모리에 공간을 할당한다.

Matrix의 소멸자에는 CPU와 GPU에 할당된 메모리를 해제한다.

```c
Matrix::Matrix(int nrow, int ncol): nrow(nrow), ncol(ncol)
{
    // 생성자
    SetCudaMem();
}
Matrix::~Matrix() {
    // 소멸자
    DelMatrix();
}
Matrix::Matrix(const Matrix& ref) 
    : nrow(ref.nrow), ncol(ref.ncol) 
{
    DelMatrix();
    SetCudaMem();
    CopyDeviceToDevice(ref);
}
```

복사생성자는 특정 클래스의 객체가 다른 객체로부터 생성될 때 호출된다. 깊은 복사(deep copy)를 수행하는 복사생성자를 직접 정의해야한다.