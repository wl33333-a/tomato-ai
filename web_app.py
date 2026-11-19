import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os

# ================= 页面配置 =================
st.set_page_config(page_title="番茄新鲜度无损预测系统", layout="centered")

# 自定义 CSS 样式：限制宽度、压缩间距、美化边框
st.markdown("""
    <style>
    /* 限制页面最大宽度，使其在宽屏上居中且紧凑 */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .main-title { font-size: 26px; font-weight: 600; color: #1E293B; border-bottom: 2px solid #CBD5E1; padding-bottom: 10px; margin-bottom: 20px; }
    .section-title { font-size: 18px; font-weight: 600; color: #334155; margin-top: 10px; margin-bottom: 10px; }
    
    /* 缩减 metric 的内边距 */
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; }
    
    /* 美化提取特征的提示框 */
    .feature-box {
        background-color: #F8FAFC; 
        padding: 12px; 
        border-left: 4px solid #3B82F6; 
        border-radius: 4px;
        margin-top: 10px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">基于智能膜的番茄新鲜度无损预测系统</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size: 14px;'>系统说明：上传智能膜图像，系统将自动提取颜色特征，并结合机器学习模型，反推番茄的预期颜色（Lab）、环境数据及新鲜度分类。</p>", unsafe_allow_html=True)

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

# ================= 界面布局 =================
st.divider()

# 输入区：左侧上传，右侧按钮
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
                <b>膜颜色特征提取结果：</b><br>
                L* = {m_lab[0]} &nbsp;|&nbsp; a* = {m_lab[1]} &nbsp;|&nbsp; b* = {m_lab[2]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("图像解析失败，请确保图像清晰且包含智能膜。")

with col2:
    st.markdown('<div class="section-title">2. 模型预测操作</div>', unsafe_allow_html=True)
    st.write("点击下方按钮，系统将运行随机森林模型进行多目标反推预测。")
    
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
                
                pred_reg = reg_model.predict(input_data)[0]
                prediction = class_model.predict(input_data)[0]
                probabilities = class_model.predict_proba(input_data)[0]
                probs = dict(zip(class_model.classes_, probabilities))
                
                st.session_state['results'] = {
                    'pred_L': round(pred_reg[0], 2),
                    'pred_a': round(pred_reg[1], 2),
                    'pred_b': round(pred_reg[2], 2),
                    'pred_co2': round(pred_reg[3], 2),
                    'pred_temp': round(pred_reg[4], 2),
                    'pred_hum': round(pred_reg[5], 2),
                    'prediction': prediction,
                    'probs': probs
                }
            else:
                st.error("无法提取膜颜色。")

# ================= 结果展示区 =================
if 'results' in st.session_state:
    res = st.session_state['results']
    
    st.markdown('<div class="section-title">3. 预测分析报告</div>', unsafe_allow_html=True)
    
    # 将数据和结论放在一个大的紧凑边框内
    with st.container(border=True):
        # 第一行：番茄数据 & 环境数据
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("**番茄预期颜色（Lab 参数）**")
            c1, c2, c3 = st.columns(3)
            c1.metric("L* (亮度)", res['pred_L'])
            c2.metric("a* (红绿)", res['pred_a'])
            c3.metric("b* (黄蓝)", res['pred_b'])
            
        with row1_col2:
            st.markdown("**反推环境数据**")
            c4, c5, c6 = st.columns(3)
            c4.metric("CO2浓度 (%)", res['pred_co2'])
            c5.metric("温度 (℃)", res['pred_temp'])
            c6.metric("湿度 (%)", res['pred_hum'])
            
        st.divider() # 内部细分割线
        
        # 第二行：新鲜度结论 & 概率
        row2_col1, row2_col2 = st.columns([1, 2])
        
        with row2_col1:
            st.markdown("**新鲜度分类评估**")
            if res['prediction'] == "新鲜":
                color = "green"
                status_text = "状态良好，符合新鲜标准。"
            elif res['prediction'] == "可接受":
                color = "#D97706"
                status_text = "轻微衰变，建议尽快食用。"
            else:
                color = "red"
                status_text = "已腐败，请勿食用。"
                
            st.markdown(f"系统判定：<span style='color:{color}; font-size: 22px; font-weight: bold;'>{res['prediction']}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#64748B; font-size: 13px;'>{status_text}</span>", unsafe_allow_html=True)
            
        with row2_col2:
            st.markdown("**分类置信度概率分布**")
            # 用更紧凑的格式显示进度条
            for state, prob in res['probs'].items():
                st.progress(float(prob), text=f"{state}: {prob*100:.2f}%")
            
else:
    st.info("等待执行预测分析... 请先上传图像并点击上方的“运行模型预测”按钮。")
