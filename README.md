# 📌 Modulabs_RecommendationSystem

딥러닝 기반 영화 추천 시스템 구현 프로젝트입니다.  
Movielens 1M 데이터셋을 기반으로 다양한 딥러닝 모델을 실험하고 성능을 비교했습니다.

---

## 🔍 프로젝트 개요

- **목표**: 사용자의 영화 평가 이력을 바탕으로 개인화된 영화 추천 시스템 개발
- **데이터**: [Movielens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- **프레임워크 및 라이브러리**: PyTorch, Pandas, NumPy, Scikit-learn 등

---

## 🗂️ 파일 구조

```bash
📦Modulabs_RecommendationSystem
 ┣ 📂data              # 전처리된 movielens 데이터
 ┣ 📂models            # 각 추천 알고리즘 구현 파일
 ┣ 📜autoint.py              # AutoInt 모델
 ┣ 📜autointimp.py           # AutoInt + MLP 결합 모델
 ┣ 📜plus_model_train.ipynb  # 모델 학습 노트북
 ┣ 📜model_load_test.ipynb   # 모델 불러오기 및 테스트
 ┣ 📜show_st.py              # Streamlit 결과 시각화
 ┗ 📜show_st2.py             # Streamlit 결과 시각화 (버전2)
```
## 🧠 구현한 모델

| 모델명 | 설명 |
|--------|------|
| `autoint.py` | AutoInt 모델 구현 (Self-Attention 기반 Feature Interaction) |
| `autointmlp.py` | AutoInt에 MLP를 결합한 모델 (AutoInt + MLP) |
| `plus_model_train.ipynb` | 개선된 모델 학습용 노트북 |
| `model_load_test.ipynb` | 학습된 모델 불러오기 및 결과 확인 |
| `show_st.py`, `show_st2.py` | Streamlit 기반 추천 결과 출력 (UI 테스트용) |

---

# 🎯 AutoInt 기반 추천 시스템 실험 결과 정리

## ✅ 모델 정보

- **모델 구조**: AutoIntMLPModel (TensorFlow 기반)
- **평가 지표**:  
  - NDCG (Normalized Discounted Cumulative Gain)  
  - Hitrate
- **목표**: 하이퍼파라미터 및 옵티마이저 변경을 통해 기본 설정보다 높은 추천 정확도 달성

---
### 평가지표 요약
| 지표          | 특징                | 고려 요소     | 값의 범위              | 해석               |
| ----------- | ----------------- | --------- | ------------------ | ---------------- |
| **HitRate** | 정답 포함 여부만 체크      | 아이템 존재 여부 | 0 또는 1 (평균하면 0\~1) | 하나라도 맞으면 성공      |
| **NDCG**    | 정답의 **순서와 위치** 고려 | 순위, 관련성   | 0\~1               | 정답이 상위일수록 가중치 부여 |
- HitRate는 "추천한 항목 중 하나라도 맞았나?"를 보는 단순 정확도.
- NDCG는 "맞긴 맞았는데 상위에 추천했나?"를 보는 정렬 품질 지표.
이 두 지표를 함께 보면,
- HitRate로는 맞췄는지 여부를,
- NDCG로는 얼마나 잘 정렬했는지를 확인할 수 있습니다.

---

## 📊 실험 결과 (평가 지표: NDCG, Hitrate)

| 실험 번호 | Description | epochs | learning_rate | dropout | batch_size | embed_dim | NDCG | Hitrate | 실험 결과 |
|-----------|-------------|--------|----------------|---------|-------------|------------|--------|----------|------------|
| 기본값 | 기본 설정 | 5 | 1e-4 | 0.4 | 2048 | 16 | **0.66302** | **0.63331** | ➖ |
| 실험1 | epochs 5 → 20 | 20 | 1e-4 | 0.4 | 2048 | 16 | 0.66250 | 0.63320 | ⬇ |
| 실험2 | epochs 20 → 10 | 10 | 1e-4 | 0.4 | 2048 | 16 | 0.66272 | 0.63330 | ⬇ |
| 실험3 | embed_dim 16 → 32 | 5 | 1e-4 | 0.4 | 2048 | 32 | 0.66240 | 0.63287 | ⬇ |
| 실험4 | learning rate 1e-4 → 5e-5 | 5 | 5e-5 | 0.4 | 2048 | 32 | 0.66260 | 0.63288 | ⬇ |
| 실험5 | embed_dim 32 → 64 | 5 | 5e-5 | 0.4 | 2048 | 64 | 0.66230 | 0.63300 | ⬇ |
| 실험6 | dropout 0.4 → 0.2 (과적합 방지 완화) | 5 | 1e-4 | 0.2 | 2048 | 32 | 0.66237 | 0.63281 | ⬇ |
| 실험7 | batch_size 2048 → 1024 | 5 | 1e-4 | 0.4 | 1024 | 32 | 0.66259 | 0.63282 | ⬇ |
| 실험8 | embed_dim=64, epochs=15 (복잡 모델 충분히 학습) | 15 | 5e-5 | 0.4 | 2048 | 64 | 0.66276 | 0.63294 | ⬇ |
| 실험9 | optimizer = AdamW | 5 | 1e-4 | 0.4 | 2048 | 32 | 0.66262 | 0.63298 | ⬇ |
| 실험10 | optimizer = RMSprop | 5 | 1e-4 | 0.4 | 2048 | 32 | 0.61544 | 0.60549 | ⬇ |
| 실험11 | optimizer = Adagrad | 5 | 1e-2 | 0.4 | 2048 | 32 | 0.55568 | 0.57097 | ⬇ |
| 실험12 | optimizer = SGD + momentum | 5 | 5e-2 | 0.4 | 2048 | 32 | 0.59711 | 0.59445 | ⬇ |
| 실험13 | loss = Focal Loss (클래스 불균형 대응) | 5 | 1e-4 | 0.4 | 2048 | 32 | 0.66254 | 0.63291 | ⬇ |

---

## 📝 실험 요약

- **기본 설정이 가장 높은 성능**을 기록  
  - `NDCG: 0.66302`, `Hitrate: 0.63331`

- **하이퍼파라미터 조정**, **옵티마이저 변경**, **loss 변경** 등 다양한 실험을 진행했으나  
  → 대부분 성능이 **미세하게 하락함**

---

## 🔎 향후 개선 방향 제안

- 모델 구조 변경  
  - 예: `xDeepFM`, `DCN`, `DeepFM`, `DIN` 등  
- Loss 함수 변경  
  - 예: `BPR Loss`, `Pairwise Hinge Loss`  
- 학습 전략 개선  
  - Learning rate scheduler, warmup, early stopping  
- Feature Engineering  
  - 중요 필드만 선택하거나 field interaction 강화

---

## 🙋‍♀️ 프로젝트 소개자
- 김귀연
- Modulabs 데이터 싸이언스 4기 추천 시스템 과정 과제 중 진행한 실습 프로젝트입니다.
  - 과제명 : MainQuest11. 딥러닝 기반 추천 시스템(Project)



