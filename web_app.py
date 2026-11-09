import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="番茄状态软测量", page_icon="🍅", layout="centered")
st.title("🍅 基于智能膜的番茄无损预测系统")
st.write("👉 只需上传**膜**的照片，系统将反推番茄的颜色（Lab）及新鲜度。")

# 替换成你实际的模型路径
MODEL_PATH = "freshness_model.pkl"

# ================= 膜识别（终极版：GrabCut 智能抠图） =================
def get_membrane_lab(img_bytes):
    if not img_bytes: return None
        
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None
    
    h, w = img.shape[:2]
    
    # 1. 定义一个中心矩形区域（告诉算法：膜大概在这个框里）
    rect_x, rect_y = int(w * 0.1), int(h * 0.1)
    rect_w, rect_h = int(w * 0.8), int(h * 0.8)
    rect = (rect_x, rect_y, rect_w, rect_h)
    
    # 2. 初始化 GrabCut 需要的掩码和模型
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 3. 运行 GrabCut 算法
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # 将 GrabCut 的结果转换成前景（膜）的二值掩码
        precise_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    except Exception as e:
        # 万一算法失败，退回中心截取（保底）
        center_img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        rgb = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = np.mean(lab[:, :, 0]) * (100.0 / 255.0)
        a = np.mean(lab[:, :, 1]) - 128.0
        b = np.mean(lab[:, :, 2]) - 128.0
        return round(L, 2), round(a, 2), round(b, 2)

    # 4. 计算 Lab 值
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_img[precise_mask > 0]
    
    if len(lab_pixels) == 0: return None
    
    L = np.mean(lab_pixels[:, 0]) * (100.0 / 255.0)
    a = np.mean(lab_pixels[:, 1]) - 128.0
    b = np.mean(lab_pixels[:, 2]) - 128.0
    return round(L, 2), round(a, 2), round(b, 2)


# ================= 网页布局 =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎨 1. 上传膜照片")
    membrane_file = st.file_uploader("选择膜特写", type=["jpg", "jpeg", "png"])
    
    if membrane_file is not None:
        img_bytes = membrane_file.getvalue()
        st.image(img_bytes, caption="已上传膜", use_container_width=True)
        
        # 使用最新的 GrabCut 算法提取
        m_lab = get_membrane_lab(img_bytes)
        
        if m_lab:
            st.success(f"✅ 膜当前颜色提取 (已智能去背景)：L={m_lab[0]}, a={m_lab[1]}, b={m_lab[2]}")
        else:
            st.error("❌ 无法提取颜色，请更换图片重试。")

with col2:
    st.subheader("🌡️ 2. 输入环境数据")
    co2 = st.number_input("CO2 浓度 (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    temp = st.number_input("温度 (℃)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
    humidity = st.number_input("湿度 (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

st.divider()

if st.button("🚀 开始反推番茄状态", use_container_width=True):
    if membrane_file is None:
        st.warning("⚠️ 请先上传膜的照片！")
    elif not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ 未找到模型文件：{MODEL_PATH}。请先运行 train_model.py。")
    else:
        m_lab = get_membrane_lab(img_bytes)
        
        if m_lab:
            models = joblib.load(MODEL_PATH)
            lab_model = models['lab_model']
            class_model = models['class_model']
            
            input_data = pd.DataFrame([{
                '膜_L': m_lab[0], '膜_a': m_lab[1], '膜_b': m_lab[2], 
                'CO2浓度': co2, '温度': temp, '湿度': humidity
            }])
            
            # 1. 预测番茄的 Lab 值
            pred_lab = lab_model.predict(input_data)[0]
            pred_L, pred_a, pred_b = round(pred_lab[0], 2), round(pred_lab[1], 2), round(pred_lab[2], 2)
            
            # 2. 预测新鲜度
            prediction = class_model.predict(input_data)[0]
            probabilities = class_model.predict_proba(input_data)[0]
            probs = dict(zip(class_model.classes_, probabilities))
            
            st.subheader("📊 软测量结果")
            
            st.info(f"🍅 **系统推断，该环境下番茄当前的预期 Lab 值为：**\n\n"
                    f"- **L (亮度)**：{pred_L}\n"
                    f"- **a (红绿)**：{pred_a}\n"
                    f"- **b (黄蓝)**：{pred_b}")
            
            if prediction == "新鲜":
                st.success(f"### 🍅 新鲜度预测：【 {prediction} 】，可以放心食用！")
            elif prediction == "可接受":
                st.warning(f"### 🍅 新鲜度预测：【 {prediction} 】，建议尽快食用。")
            else:
                st.error(f"### 🍅 新鲜度预测：【 {prediction} 】，请勿食用！")
            
            st.write("各状态的概率分布如下：")
            for state, prob in probs.items():
                st.progress(float(prob), text=f"{state}: {prob*100:.1f}%")