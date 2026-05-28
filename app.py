import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

st.title("폐 건강 군집화 프로그램")

# CSV 파일 불러오기
df = pd.read_csv('lung_data.csv')

# 사용할 데이터
X = df[['Smokes', 'Age', 'Alkhol']]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans 모델 생성
model = KMeans(n_clusters=3, random_state=42)
model.fit(X_scaled)

# 군집 번호 저장
df['cluster'] = model.labels_

# 모델 저장
joblib.dump(model, 'lung_model.pkl')
joblib.dump(scaler, 'lung.scaler.pkl')

# 사용자 입력
st.sidebar.header("환자 정보 입력")

Smokes = st.sidebar.number_input("담배", min_value=0.0)
Age = st.sidebar.number_input("나이", min_value=0.0)
Alkhol = st.sidebar.number_input("알코올", min_value=0.0)

# 예측 버튼
if st.button("군집 예측하기"):

    # 새 환자 데이터 생성
    new_patient = pd.DataFrame(
        [[Smokes, Age, Alkhol]],
        columns=['Smokes', 'Age', 'Alkhol']
    )

    # 표준화
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.subheader("예측 결과")
    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        df['Smokes'],
        df['Age'],
        c=df['cluster'],
        alpha=0.5
    )

    # 새 환자 표시
    ax.scatter(
        Smokes,
        Age,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel('Smokes')
    ax.set_ylabel('Age')
    ax.legend()

    # Streamlit에 그래프 출력
    st.pyplot(fig)

# 데이터 미리보기
st.subheader("데이터 확인")
st.dataframe(df.head())