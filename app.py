import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# 1. 한글 폰트 설정 (코드 최상단 유지)
# =========================================================
def setup_korean_font():
    # Streamlit Cloud(리눅스) 환경의 나눔 폰트 경로
    linux_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    if os.path.exists(linux_font_path):
        # 서버 환경 (packages.txt 설치 필요)
        font_prop = fm.FontProperties(fname=linux_font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        # 로컬 환경 (Windows/macOS)
        plt.rc('font', family='Malgun Gothic') # 윈도우 기준
        
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# =========================================================
# 2. 메인 앱 로직
# =========================================================
st.title("🫁 폐 건강 군집화 프로그램")

# CSV 파일 불러오기
try:
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

except FileNotFoundError:
    st.error("파일을 찾을 수 없습니다. 'lung_data.csv'가 경로에 있는지 확인해주세요.")
    st.stop()

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

    # 표준화 및 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.subheader("📊 예측 결과")
    st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 그래프 생성 (이제 한글이 깨지지 않습니다!)
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df['Smokes'],
        df['Age'],
        c=df['cluster'],
        alpha=0.5,
        cmap='viridis'
    )

    # 새 환자 표시
    ax.scatter(
        Smokes,
        Age,
        c='red',
        s=300,
        marker='X',
        label='현재 환자 위치'
    )

    ax.set_xlabel('흡연량 (Smokes)')
    ax.set_ylabel('나이 (Age)')
    ax.set_title('폐 건강 데이터 군집 시각화')
    ax.legend()

    # Streamlit에 그래프 출력
    st.pyplot(fig)

# 데이터 미리보기
st.divider()
st.subheader("📂 데이터 확인 (상위 5행)")
st.dataframe(df.head())
