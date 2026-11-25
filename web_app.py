import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os

# ================= 页面配置 =================
st.set_page_config(page_title="番茄生理状态无损监测", layout="centered")

# 科研风紧凑 CSS
st.markdown("""
    <style>
    .block-container { max-width: 900px; padding-top: 4rem !important; padding-bottom: 2rem; }
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .main-title { font-size: 28px; font-weight: 600; color: #1E293B; line-height: 1.5 !important; padding-bottom: 12px; border-bottom: 2px solid #CBD5E1; margin-bottom: 20px; }
    .section-title { font-size: 18px; font-weight: 600; color: #334155; margin-top: 10px; margin-bottom: 10px; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #0F172A; }
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #64748B; }
    .feature-box { background-color: #F8FAFC; padding: 12px; border-left: 4px solid #3B82F6; border-radius: 4px; margin-top: 10px; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">基于智能膜的番茄生理状态无损监测系统</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size: 14px;'>系统说明：上传智能膜图像，系统将提取颜色特征，并通过机器学习模型反推番茄当前的呼吸速率与新鲜度等级。</p>", unsafe_allow_html=True)

MODEL_PATH = "freshness_model.pkl"

# ================= 颜色提取核心函数 =================
def get_membrane_lab(img_bytes):
    if not img_bytes: return None
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None
    
    h, w = img.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        precise_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    except:
        precise_mask = np.ones((h, w), dtype=np.uint8) * 255

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_img[precise_mask > 0]
    if len(lab_pixels) == 0: return None
    
    L = np.mean(lab_pixels[:, 0]) * (100.0 / 255.0)
    a = np.mean(lab_pixels[:, 1]) - 128.0
    b = np.mean(lab_pixels[:, 2]) - 128.0
    return round(L, 2), round(a, 2), round(b, 2)

# ================= 输入区 =================
st.divider()
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="section-title">1. 图像数据输入</div>', unsafe_allow_html=True)
    membrane_file = st.file_uploader("请上传智能膜图像", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if membrane_file is not None:
        img_bytes = membrane_file.getvalue()
        st.image(img_bytes, caption="已加载样本图像", use_container_width=True)
        
        m_lab = get_membrane_lab(img_bytes)
        if m_lab:
            st.markdown(f"""
            <div class="feature-box">
                <b>膜颜色特征提取：</b><br>
                L* = {m_lab[0]} &nbsp;|&nbsp; a* = {m_lab[1]} &nbsp;|&nbsp; b* = {m_lab[2]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("图像解析失败，请确保图像清晰。")

with col2:
    st.markdown('<div class="section-title">2. 模型预测操作</div>', unsafe_allow_html=True)
    st.write("点击下方按钮，系统将根据膜的颜色反推番茄生理状态。")
    st.write("") 
    
    if st.button("运行模型预测 (Run Prediction)", use_container_width=True, type="primary"):
        if membrane_file is None:
            st.error("请先上传智能膜图像。")
        elif not os.path.exists(MODEL_PATH):
            st.error(f"未找到模型文件：{MODEL_PATH}，请先训练模型。")
        else:
            m_lab = get_membrane_lab(img_bytes)
            if m_lab:
                models = joblib.load(MODEL_PATH)
                reg_model = models['reg_model'] 
                class_model = models['class_model']
                
                input_data = pd.DataFrame([{
                    '膜_L': m_lab[0], '膜_a': m_lab[1], '膜_b': m_lab[2]
                }])
                
                # 【核心修复】将预测结果强制转换为浮点数，避免 array 导致 round 报错
                pred_rr_raw = reg_model.predict(input_data)
                pred_rr = float(pred_rr_raw[0])
                
                prediction = class_model.predict(input_data)[0]
                probabilities = class_model.predict_proba(input_data)[0]
                probs = dict(zip(class_model.classes_, probabilities))
                
                st.session_state['results'] = {
                    'pred_rr': round(pred_rr, 2),
                    'prediction': prediction,
                    'probs': probs
                }

# ================= 结果展示区 =================
st.divider()
st.markdown('<div class="section-title">3. 预测分析报告</div>', unsafe_allow_html=True)

if 'results' in st.session_state:
    res = st.session_state['results']
else:
    res = {
        'pred_rr': "--",
        'prediction': "待预测",
        'probs': {"新鲜": 0.0, "可接受": 0.0, "腐败": 0.0}
    }

# 紧凑的仪表盘面板
with st.container(border=True):
    # 第一行：核心指标（呼吸速率 + 新鲜度）
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("**番茄呼吸速率预测**")
        st.metric(label="呼吸速率 (mL CO₂ / kg·h)", value=res['pred_rr'])
        
    with row1_col2:
        st.markdown("**新鲜度分类评估**")
        if res['prediction'] == "待预测":
            st.markdown(f"<span style='color:#64748B; font-size: 24px; font-weight: bold;'>待预测</span>", unsafe_allow_html=True)
        else:
            if res['prediction'] == "新鲜":
                color = "#10B981" # 科研常用的绿色
            elif res['prediction'] == "可接受":
                color = "#F59E0B" # 琥珀色
            else:
                color = "#EF4444" # 红色
            st.markdown(f"<span style='color:{color}; font-size: 24px; font-weight: bold;'>{res['prediction']}</span>", unsafe_allow_html=True)
        
    st.divider() 
    
    # 第二行：分类概率分布
    st.markdown("**新鲜度判定置信度**")
    for state, prob in res['probs'].items():
        st.progress(float(prob), text=f"{state}: {prob*100:.2f}%")
