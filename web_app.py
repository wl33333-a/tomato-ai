import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="番茄状态软测量", page_icon="🍅", layout="centered")
st.title("🍅 基于智能膜的番茄无损预测系统")
st.write("👉 只需上传**膜**的照片，系统将全自动反推番茄的颜色、新鲜度以及当前环境数据。")

MODEL_PATH = "freshness_model.pkl" # 云端相对路径

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
        precise_mask = np.ones((h, w), dtype=np.uint8) * 255 # 失败则全图

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_img[precise_mask > 0]
    if len(lab_pixels) == 0: return None
    
    L = np.mean(lab_pixels[:, 0]) * (100.0 / 255.0)
    a = np.mean(lab_pixels[:, 1]) - 128.0
    b = np.mean(lab_pixels[:, 2]) - 128.0
    return round(L, 2), round(a, 2), round(b, 2)

# 只需要上传
membrane_file = st.file_uploader("选择膜特写", type=["jpg", "jpeg", "png"])

if membrane_file is not None:
    img_bytes = membrane_file.getvalue()
    st.image(img_bytes, caption="已上传膜", width=300)
    
    m_lab = get_membrane_lab(img_bytes)
    if m_lab:
        st.success(f"✅ 膜当前颜色：L={m_lab[0]}, a={m_lab[1]}, b={m_lab[2]}")
        
        if st.button("🚀 开始全自动推算", use_container_width=True):
            if not os.path.exists(MODEL_PATH):
                st.error("⚠️ 未找到模型文件！")
            else:
                models = joblib.load(MODEL_PATH)
                
                # 输入只有膜
                input_data = pd.DataFrame([{'膜_L': m_lab[0], '膜_a': m_lab[1], '膜_b': m_lab[2]}])
                
                # 1. 预测连续变量
                pred_reg = models['reg_model'].predict(input_data)[0]
                pred_t_L, pred_t_a, pred_t_b = round(pred_reg[0], 2), round(pred_reg[1], 2), round(pred_reg[2], 2)
                pred_co2, pred_temp, pred_hum = round(pred_reg[3], 2), round(pred_reg[4], 2), round(pred_reg[5], 2)
                
                # 2. 预测新鲜度
                prediction = models['class_model'].predict(input_data)[0]
                probs = models['class_model'].predict_proba(input_data)[0]
                prob_dict = dict(zip(models['class_model'].classes_, probs))
                
                st.divider()
                st.subheader("📊 全自动推算结果")
                
                # 展示结果
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🍅 **番茄预期 Lab**\n\n"
                            f"L: {pred_t_L}\n\n a: {pred_t_a}\n\n b: {pred_t_b}")
                with col2:
                    st.warning(f"🌡️ **反推环境数据**\n\n"
                               f"CO2: {pred_co2} %\n\n 温度: {pred_temp} ℃\n\n 湿度: {pred_hum} %")
                
                if prediction == "新鲜":
                    st.success(f"### 🍅 新鲜度预测：【 {prediction} 】，可以放心食用！")
                elif prediction == "可接受":
                    st.warning(f"### 🍅 新鲜度预测：【 {prediction} 】，建议尽快食用。")
                else:
                    st.error(f"### 🍅 新鲜度预测：【 {prediction} 】，请勿食用！")
                
                for state, prob in prob_dict.items():
                    st.progress(float(prob), text=f"{state}: {prob*100:.1f}%")
