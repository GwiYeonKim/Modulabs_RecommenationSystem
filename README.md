# 📌 Modulabs_RecommendationSystem

딥러닝 기반 영화 추천 시스템 구현 프로젝트입니다.  
Movielens 1M 데이터셋을 기반으로 다양한 딥러닝 모델을 실험하고 성능을 비교했습니다.

---

## 🔍 프로젝트 개요

- **목표**: 사용자의 영화 평가 이력을 바탕으로 개인화된 영화 추천 시스템 개발
- **데이터**: [Movielens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- **프레임워크 및 라이브러리**: PyTorch, Pandas, NumPy, Scikit-learn 등

---

## 🧠 구현한 모델

| 모델명 | 설명 |
|--------|------|
| `autoint.py` | AutoInt 모델 구현 (Self-Attention 기반 Feature Interaction) |
| `autointimp.py` | AutoInt에 MLP를 결합한 모델 (AutoInt + MLP) |
| `plus_model_train.ipynb` | 개선된 모델 학습용 노트북 |
| `model_load_test.ipynb` | 학습된 모델 불러오기 및 결과 확인 |
| `show_st.py`, `show_st2.py` | Streamlit 기반 추천 결과 출력 (UI 테스트용) |

---

## 🗂️ 파일 구조

```bash
📦Modulabs_RecommendationSystem
 ┣ 📜autoint.py              # AutoInt 모델
 ┣ 📜autointimp.py           # AutoInt + MLP 결합 모델
 ┣ 📜plus_model_train.ipynb  # 모델 학습 노트북
 ┣ 📜model_load_test.ipynb   # 모델 불러오기 및 테스트
 ┣ 📜show_st.py              # Streamlit 결과 시각화
 ┗ 📜show_st2.py             # Streamlit 결과 시각화 (버전2)


## 📊 실험 결과 (평가 지표: Hit@10, nDCG@10)
| 모델명           | Hit\@10   | nDCG\@10  |
| ------------- | --------- | --------- |
| MF            | 0.612     | 0.438     |
| NeuralCF      | 0.644     | 0.459     |
| AutoInt       | 0.681     | 0.488     |
| AutoInt + MLP | **0.694** | **0.502** |

> 📌 AutoInt + MLP 모델이 모든 지표에서 가장 우수한 성능을 기록했습니다.

## 🙋‍♀️ 프로젝트 소개자
- 김귀연
- Modulabs 데이터 싸이언스 4기 추천 시스템 과정 과제 중 진행한 실습 프로젝트입니다.
  - 과제명 : MainQuest11. 딥러닝 기반 추천 시스템(Project)


