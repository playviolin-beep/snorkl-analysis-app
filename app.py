import streamlit as st
import pandas as pd
import os
import glob
import google.generativeai as genai
import time
import altair as alt
import re

# ==========================================
# 1. 설정 및 유틸리티
# ==========================================
st.set_page_config(page_title="Snorkl 마스터 (최종)", layout="wide")
st.title("🏫 Snorkl 학급별 세특 관리 시스템 (최종 완성판)")

# 필수 폴더 생성
for folder in ["data", "data/classes", "data/history", "data/questions"]:
    if not os.path.exists(folder): os.makedirs(folder)

def load_csv_safe(file_buffer):
    # CSV 파일 안전 로드
    try:
        return pd.read_csv(file_buffer, encoding='utf-8')
    except UnicodeDecodeError:
        file_buffer.seek(0)
        return pd.read_csv(file_buffer, encoding='cp949')
    except Exception: return None

def get_grade_char(score):
    # 4점 척도 변환
    try:
        s = float(score)
        if s >= 4: return "A"
        elif s >= 3: return "B"
        elif s >= 2: return "C"
        elif s >= 1: return "D"
        else: return "E"
    except: return "E"

# [정답 판단 로직]
def check_is_correct(value):
    val_str = str(value).lower().strip()
    true_keywords = ['yes', 'true', 'pass', 'correct', 'right', 'o', '정답', '맞음', 'y', 't']
    
    if val_str in true_keywords: return 1
    for k in true_keywords:
        if val_str.startswith(k): return 1
    return 0

# [핵심 기능] Snorkl 데이터 구조 변환
def process_snorkl_data(df):
    long_data = []
    cols = df.columns
    prefix_map = {} 
    
    for c in cols:
        if "Response" in c and "Best" not in c:
            match = re.match(r"^(\d+)(st|nd|rd|th)", c)
            if match:
                order_num = int(match.group(1))
                prefix = match.group(0) 
                if prefix + " Response" in c:
                    prefix_map[order_num] = prefix + " Response"

    sorted_orders = sorted(prefix_map.keys())
    
    for idx, row in df.iterrows():
        f_name = str(row.get('First Name', '')).strip()
        l_name = str(row.get('Last Name', '')).strip()
        
        for order in sorted_orders:
            prefix = prefix_map[order]
            col_correct = next((c for c in cols if c.startswith(prefix) and ("Correct" in c or "Pass" in c)), None)
            col_score = next((c for c in cols if c.startswith(prefix) and "Score" in c), None)
            
            if col_correct and col_score:
                val_correct = row[col_correct]
                val_score = row[col_score]
                
                if str(val_correct).strip() in ['-', 'nan', '', 'None', 'NaT']: continue
                
                try: num_score = float(val_score)
                except: num_score = 0.0
                
                long_data.append({
                    'First Name': f_name,
                    'Last Name': l_name,
                    'Attempt_Order': order,
                    'Raw_Correct': str(val_correct),
                    'Numeric_Score': num_score,
                    'Is_Correct_Val': check_is_correct(val_correct)
                })
                
    return pd.DataFrame(long_data)

# [데이터 로드 및 컬럼 표준화]
def standardize_columns(df):
    if 'Q_Title' in df.columns:
        df.rename(columns={'Q_Title': 'Question_Title'}, inplace=True)
    return df

# ==========================================
# 2. 데이터 로드
# ==========================================
RUBRIC_DB = {} 
rubric_path = os.path.join("data", "rubric.csv")
if os.path.exists(rubric_path):
    try:
        df_r = pd.read_csv(rubric_path)
        RUBRIC_DB = df_r.set_index('성취기준').to_dict('index')
    except: pass

# ==========================================
# 3. 사이드바
# ==========================================
with st.sidebar:
    st.header("🔧 관리자 메뉴")
    api_key = st.text_input("Google AI Studio Key", type="password")
    
    st.divider()
    st.subheader("📂 데이터 파일 관리")
    up_rubric = st.file_uploader("1. 성취기준 매핑용 (rubric.csv)", type='csv')
    if up_rubric:
        with open(rubric_path, "wb") as f: f.write(up_rubric.getbuffer())
        st.success("저장 완료"); time.sleep(1); st.rerun()
    
    st.info(f"DB 현황: 성취기준 {len(RUBRIC_DB)}개")

    st.divider()
    st.subheader("📂 수업(반) 등록")
    new_class_name = st.text_input("수업명 (예: 1학년7반)")
    roster_file = st.file_uploader("명렬표 CSV", type=['csv'], key="roster")
    if st.button("수업 등록"):
        if new_class_name and roster_file:
            df = load_csv_safe(roster_file)
            if df is not None:
                df = df.astype(str)
                for c in df.columns: df[c] = df[c].str.strip()
                df.to_csv(os.path.join("data", "classes", f"roster_{new_class_name}.csv"), index=False, encoding='utf-8-sig')
                st.success("등록 완료")

    def get_class_list():
        files = glob.glob(os.path.join("data", "classes", "roster_*.csv"))
        return [os.path.basename(f).replace("roster_", "").replace(".csv", "") for f in files]

# ==========================================
# 4. 메인 화면
# ==========================================
class_list = get_class_list()

if not class_list:
    st.warning("👈 왼쪽 사이드바에서 수업을 먼저 등록해주세요.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📝 1. 문항 등록", "📤 2. 결과 누적", "🧐 3. 팩트 추출", "📊 4. 통계 대시보드"])

    # [탭 1] 문항 등록 (버그 수정됨)
    with tab1:
        st.subheader("📌 문항 등록 (AI 분석의 기준)")
        c1, c2 = st.columns([2, 1])
        with c1:
            q_title = st.text_input("문항 제목", placeholder="예: 1-1-1. 문제1")
            q_prompt = st.text_area("문항 지문 (구체적으로 입력하세요!)")
            q_answer = st.text_area("정답/평가요소 (AI가 참고할 핵심 포인트)")
        with c2:
            st.markdown("**성취기준 매핑**")
            std_opts = list(RUBRIC_DB.keys()) if RUBRIC_DB else ["데이터 없음"]
            
            # [버그 수정] Session State를 사용하여 선택 인덱스 관리
            if 'std_select_index' not in st.session_state:
                st.session_state['std_select_index'] = 0

            if st.button("🤖 AI 매핑 추천"):
                if not api_key: st.error("API 키 필요")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        res = model.generate_content(f"문제: {q_prompt}\n정답: {q_answer}\n\n다음 중 가장 적절한 성취기준 하나를 골라 그대로 출력해:\n{chr(10).join(std_opts)}")
                        rec_std = res.text.strip()
                        
                        if rec_std in std_opts:
                            # 추천된 값의 인덱스를 찾아서 Session State 업데이트
                            new_index = std_opts.index(rec_std)
                            st.session_state['std_select_index'] = new_index
                            st.success(f"추천 완료: {rec_std}")
                            time.sleep(0.5)
                            st.rerun() # 화면 새로고침하여 드롭다운 변경 적용
                        else:
                            st.warning(f"추천된 값({rec_std})이 목록에 없습니다.")
                    except: pass
            
            # index 파라미터에 Session State 연결
            q_std = st.selectbox("성취기준 선택", std_opts, index=st.session_state['std_select_index'])
            
            if st.button("문항 저장"):
                if q_title:
                    new_df = pd.DataFrame([{"Title": q_title, "Prompt": q_prompt, "Answer": q_answer, "Standard": q_std}])
                    p = "data/questions_db.csv"
                    new_df.to_csv(p, mode='a', header=not os.path.exists(p), index=False, encoding='utf-8-sig')
                    st.success("저장 완료")

    # [탭 2] 결과 누적
    with tab2:
        st.subheader("📥 Snorkl 결과 업로드")
        c1, c2 = st.columns(2)
        target_class = c1.selectbox("수업 선택", class_list)
        q_list = []
        if os.path.exists("data/questions_db.csv"):
            q_list = pd.read_csv("data/questions_db.csv")['Title'].tolist()
        sel_q = c2.selectbox("문항 선택", q_list)
        
        up_res = st.file_uploader("Snorkl CSV 업로드", type='csv')
        
        if up_res and st.button("데이터 처리 및 저장"):
            df_res = load_csv_safe(up_res)
            if df_res is not None:
                long_df = process_snorkl_data(df_res)
                if long_df.empty:
                    st.error("데이터 변환 실패: 유효한 응답이 없습니다.")
                else:
                    df_cls = pd.read_csv(os.path.join("data", "classes", f"roster_{target_class}.csv"))
                    df_cls = df_cls.astype(str)
                    for c in df_cls.columns: df_cls[c] = df_cls[c].str.strip()
                    
                    long_df['First Name'] = long_df['First Name'].astype(str).str.strip()
                    long_df['Last Name'] = long_df['Last Name'].astype(str).str.strip()
                    
                    merged = pd.merge(df_cls, long_df, on=['First Name', 'Last Name'], how='left')
                    merged['Question_Title'] = sel_q
                    
                    ts = int(time.time())
                    path = os.path.join("data", "history", f"{target_class}_{ts}.csv")
                    merged.to_csv(path, index=False, encoding='utf-8-sig')
                    with open(path.replace(".csv", "_meta.txt"), "w") as f: f.write(sel_q)
                    
                    st.success(f"저장 완료 ({len(merged)}행)")

    # [탭 3] 팩트 추출
    with tab3:
        st.subheader("🧐 데이터 기반 역량 팩트 추출")
        t_cls = st.selectbox("분석 대상 반", class_list, key="final")
        
        if st.button("데이터 로드 & 분석 준비"):
            files = glob.glob(os.path.join("data", "history", f"{t_cls}_*.csv"))
            if not files: st.warning("데이터 없음")
            else:
                full_df = pd.DataFrame()
                q_db = pd.read_csv("data/questions_db.csv")
                for fp in files:
                    try:
                        tmp = pd.read_csv(fp)
                        tmp = standardize_columns(tmp)
                        if 'Numeric_Score' not in tmp.columns:
                            tmp = process_snorkl_data(tmp)
                            tmp = standardize_columns(tmp)
                        
                        meta = fp.replace(".csv", "_meta.txt")
                        q_t = ""
                        if os.path.exists(meta):
                            with open(meta) as f: q_t = f.read().strip()
                            tmp['Question_Title'] = q_t
                        
                        if not tmp.empty and q_t:
                            qi = q_db[q_db['Title'] == q_t]
                            if not qi.empty:
                                tmp['Standard'] = qi.iloc[0]['Standard']
                                tmp['Prompt'] = qi.iloc[0]['Prompt']
                                tmp['Answer'] = qi.iloc[0]['Answer']
                            else:
                                tmp['Standard'] = "미등록 성취기준"
                                tmp['Prompt'] = "정보 없음"
                                tmp['Answer'] = "정보 없음"
                        full_df = pd.concat([full_df, tmp])
                    except: pass
                
                if 'Numeric_Score' in full_df.columns:
                    full_df = full_df.dropna(subset=['Numeric_Score'])
                    st.session_state['grouped'] = full_df.groupby(['First Name', 'Last Name'])
                    st.session_state['total_db_questions'] = len(q_db)
                    st.success(f"로드 완료 (총 {len(full_df)}건 시도)")
                else: st.error("유효 데이터 없음")

        if 'grouped' in st.session_state and st.button("🚀 팩트 리포트 생성"):
            if not api_key: st.error("API 키 필요")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                results = []
                bar = st.progress(0)
                students = list(st.session_state['grouped'])
                total_db_q = st.session_state.get('total_db_questions', 0)
                
                for idx, ((fname, lname), s_df) in enumerate(students):
                    # 정량적 데이터 계산
                    attempted_q_count = s_df['Question_Title'].nunique()
                    solved_q_count = s_df.groupby('Question_Title')['Is_Correct_Val'].max().sum()
                    perfect_score_count = s_df[s_df['Numeric_Score'] >= 4.0]['Question_Title'].nunique()
                    
                    stats_text = f"""
                    [정량적 성취 데이터]
                    - 전체 문항 수(DB 기준): {total_db_q}개
                    - 시도한 문항 수: {attempted_q_count}개
                    - 해결(정답) 문항 수: {solved_q_count}개
                    - 만점(4.0) 달성 문항 수: {perfect_score_count}개
                    """
                    
                    analysis_data = ""
                    if 'Standard' in s_df.columns:
                        for std, grp in s_df.groupby('Standard'):
                            analysis_data += f"\n=== [성취기준] {std} ===\n"
                            q_grp = grp.groupby('Question_Title')
                            for qt, q_sub in q_grp:
                                q_sub = q_sub.sort_values('Attempt_Order')
                                attempts = len(q_sub)
                                scores = q_sub['Numeric_Score'].tolist()
                                has_success = 1 in q_sub['Is_Correct_Val'].tolist()
                                start, end = scores[0], scores[-1]
                                
                                p_txt = str(q_sub.iloc[0].get('Prompt', ''))
                                a_txt = str(q_sub.iloc[0].get('Answer', ''))
                                
                                analysis_data += f"""
                                - 문항: {qt}
                                - 문제 내용: {p_txt}
                                - 평가 핵심: {a_txt}
                                - 이력: {attempts}회 시도 ({start} -> {end})
                                - 결과: {'성공(Yes)' if has_success else '실패(No)'}
                                """
                    
                    prompt = f"""
                    당신은 냉철한 교육 데이터 분석가입니다. 학생 데이터를 보고 정량적 통계와 질적 팩트를 요약하세요.

                    [입력 데이터]
                    {stats_text}
                    {analysis_data}

                    [작성 원칙]
                    1. **정량 데이터 필수**: [종합 요약]에 위 통계 수치를 그대로 적을 것.
                    2. **일관된 포맷**: 모든 학생에게 동일한 형식 적용.
                    3. **팩트 중심**: 미사여구 제외, 건조한 서술.

                    [출력 포맷 예시]
                    [종합 요약]
                    - 전체 문항: (숫자)개 / 시도: (숫자)개 / 해결: (숫자)개 / 만점: (숫자)개

                    [성취기준: (성취기준명)]
                    1. (문항명)
                       - 내용: (문제 핵심 요약)
                       - 결과: (N)회 시도 후 (성공/실패), 점수 변화((시작)->(끝))
                       - 역량: (평가 핵심에 기반한 역량 팩트)
                    
                    (반복)
                    """
                    try:
                        time.sleep(0.5)
                        res = model.generate_content(prompt)
                        out = res.text.strip()
                    except: out = "실패"
                    
                    results.append({
                        "학번": lname, "이름": fname,
                        "전체_문항": total_db_q, "시도": attempted_q_count,
                        "해결": solved_q_count, "만점": perfect_score_count,
                        "리포트": out
                    })
                    bar.progress((idx+1)/len(students))
                
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                st.download_button("💾 팩트 리포트 다운로드", res_df.to_csv(index=False, encoding='utf-8-sig'), "팩트_리포트.csv")

    # [탭 4] 대시보드
    with tab4:
        st.subheader("📊 종합 통계 대시보드")
        all_files = glob.glob(os.path.join("data", "history", "*.csv"))
        if all_files:
            df_list = []
            for f in all_files:
                try: 
                    t = pd.read_csv(f)
                    t = standardize_columns(t)
                    if 'Numeric_Score' not in t.columns:
                        t = process_snorkl_data(t)
                        t = standardize_columns(t)
                    meta = f.replace(".csv", "_meta.txt")
                    if os.path.exists(meta):
                        with open(meta) as f2: t['Question_Title'] = f2.read().strip()
                    df_list.append(t)
                except: pass
            
            if df_list:
                df_all = pd.concat(df_list, ignore_index=True)
                if 'Numeric_Score' in df_all.columns:
                    df_valid = df_all.dropna(subset=['Numeric_Score'])
                    if not df_valid.empty:
                        q_solved = df_valid.groupby(['Last Name', 'First Name', 'Question_Title'])['Is_Correct_Val'].max()
                        solve_rate = q_solved.mean() * 100
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("총 누적 시도", f"{len(df_valid)}회")
                        m2.metric("문항 해결률", f"{solve_rate:.1f}%")
                        m3.metric("평균 점수", f"{df_valid['Numeric_Score'].mean():.2f}")
                        
                        st.divider()
                        st.markdown("### 📉 개인별 이력")
                        df_valid['Key'] = df_valid['Last Name'].astype(str) + " " + df_valid['First Name'].astype(str)
                        sel = st.selectbox("학생 선택", df_valid['Key'].unique())
                        if sel:
                            s_data = df_valid[df_valid['Key'] == sel].copy().sort_values('Attempt_Order')
                            s_data['Seq'] = range(1, len(s_data)+1)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.altair_chart(alt.Chart(s_data).mark_line(point=True).encode(
                                    x=alt.X('Seq:O', title='순서'), y='Numeric_Score', color='Question_Title', tooltip=['Question_Title', 'Numeric_Score']
                                ), use_container_width=True)
                            with c2:
                                st.altair_chart(alt.Chart(s_data).mark_circle(size=100).encode(
                                    x=alt.X('Seq:O', title='순서'), y='Is_Correct_Val', color='Is_Correct_Val', tooltip=['Question_Title', 'Is_Correct_Val']
                                ), use_container_width=True)
                else: st.info("유효 데이터 없음")
        else: st.info("데이터 파일 없음")