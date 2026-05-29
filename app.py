import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. 한글 폰트 및 환경 설정 (최상단 유지)
# ==========================================
def unique_config_font():
    # Streamlit Cloud 환경 (리눅스) 폰트 경로
    path_linux = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    if os.path.exists(path_linux):
        # 운영 서버 환경 (packages.txt에 fonts-nanum 추가 필수)
        font_name = fm.FontProperties(fname=path_linux).get_name()
        plt.rc('font', family=font_name)
    else:
        # 로컬 환경 (Windows/macOS)
        # Windows는 'Malgun Gothic', macOS는 'AppleGothic'이 기본입니다.
        plt.rc('font', family='Malgun Gothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

unique_config_font()

# ==========================================
# 2. 데이터 로드 및 모델링
# ==========================================
st.title("🫁 폐 건강 군집화 프로그램")

# CSV 파일 불러오기 (파일이 같은 경로에 있어야 합니다)
try:
    df = pd.read_csv('lung_data.csv')

    # 사용할 데이터
    X = df[['Smokes', 'Age', 'Alkhol']]

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans 모델 생성
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_scaled)

    # 군집 번호 저장
    df['cluster'] = model.labels_

    # 모델 저장 (캐싱을 위해 실무에선 st.cache_data 사용 권장)
    joblib.dump(model, 'lung_model.pkl')
    joblib.dump(scaler, 'lung_scaler.pkl')

except FileNotFoundError:
    st.error("파일을 찾을 수 없습니다. 'lung_data.csv' 파일이 저장소에 있는지 확인해주세요.")
    st.stop()

# ==========================================
# 3. 사이드바 - 사용자 입력
# ==========================================
st.sidebar.header("환자 정보 입력")

Smokes = st.sidebar.number_input("담배 (Smokes)", min_value=0.0, value=0.0)
Age = st.sidebar.number_input("나이 (Age)", min_value=0.0, value=30.0)
Alkhol = st.sidebar.number_input("알코올 (Alkhol)", min_value=0.0, value=0.0)

# ==========================================
# 4. 예측 및 시각화
# ==========================================
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
    st.subheader("📊 예측 결과")
    st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # 전체 데이터 산점도
    scatter = ax.scatter(
        df['Smokes'],
        df['Age'],
        c=df['cluster'],
        cmap='viridis',
        alpha=0.5
    )

    # 새 환자 위치 표시 (범례에 한글이 나옵니다)
    ax.scatter(
        Smokes,
        Age,
        c='red', # 눈에 띄게 빨간색으로 변경
        s=200,
        marker='X',
        label='예측 환자 위치'
    )

    ax.set_xlabel('흡연량 (Smokes)')
    ax.set_ylabel('나이 (Age)')
    ax.set_title('폐 건강 군집 분석 결과')
    ax.legend()

    # Streamlit에 그래프 출력
    st.pyplot(fig)

# 데이터 미리보기
st.divider()
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head())
