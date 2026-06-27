## MNIST Classification with Custom Cross-Entropy Loss

PyTorch를 사용하여 MNIST 수기 숫자 데이터셋을 분류하는 Convolutional Neural Network (CNN) 모델입니다. 이 프로젝트의 핵심은 PyTorch의 내장 손실 함수인 `nn.CrossEntropyLoss`를 사용하지 않고, 오버플로우(Overflow)를 방지하는 커스텀 교차 엔트로피 손실 함수(Custom Cross-Entropy Loss)를 직접 구현하여 적용하였습니다.

---

### ✨ Key Features

* **Custom Loss Function:** Log-Sum-Exp 트릭을 적용하여 수치적으로 안정적인 Cross-Entropy Loss를 직접 구현했습니다.
* **CNN Architecture:** 3개의 합성곱 계층(Convolution Layer)과 과적합 방지를 위한 드롭아웃(Dropout)이 적용된 완전 연결 계층(Fully Connected Layer)으로 구성되었습니다.
* **Visualization:** Matplotlib을 활용해 매 에포크(Epoch)마다 학습 및 테스트 데이터에 대한 손실(Loss)과 정확도(Accuracy) 추이를 시각화합니다.

---

### 🧠 Model Architecture

* **Conv Block 1:** Conv2D (1 to 32 channels, kernel 5x5, padding 'same') + ReLU + MaxPool2D (2x2)
* **Conv Block 2:** Conv2D (32 to 64 channels, kernel 5x5, padding 'same') + ReLU + MaxPool2D (2x2)
* **Conv Block 3:** Conv2D (64 to 128 channels, kernel 5x5, padding 'same') + ReLU + MaxPool2D (2x2)
* **Classifier:** Flatten + Linear (1024 nodes) + Dropout (0.5) + Linear (10 nodes)

---

### 📐 Custom Cross-Entropy Loss Implementation

단순히 소프트맥스(Softmax)와 로그를 취할 경우 지수 연산에서 발생할 수 있는 값의 폭발을 막기 위해, 로짓(Logits)의 최댓값을 빼주는 방식으로 수식을 구현했습니다.

$$L = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(x_{i, y_i} - \max_j x_{i,j})}{\sum_k \exp(x_{i, k} - \max_j x_{i,j})} \right)$$

---

### 📊 Training & Results

* **Optimizer:** Adam
* **Learning Rate:** 1e-4
* **Batch Size:** 32

학습 결과, 테스트 데이터셋에서 약 99.3%의 높은 정확도(Accuracy)를 달성했습니다. 학습이 완료되면 자동으로 `training_result.png` 파일이 생성되어 모델의 수렴 과정을 직관적으로 확인할 수 있습니다.

---

### 🚀 How to Run

1. 필요한 라이브러리를 설치합니다. (`torch`, `torchvision`, `matplotlib`)
2. 스크립트를 실행하여 모델 학습을 시작합니다.
3. 학습이 완료되면 콘솔에서 최종 정확도를 확인하고, 저장된 시각화 그래프를 확인합니다.
