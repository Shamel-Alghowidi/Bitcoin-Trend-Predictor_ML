import streamlit as st
import joblib
import numpy as np

# 1. إعداد عنوان الصفحة وشكلها
st.set_page_config(page_title="Bitcoin Predictor", page_icon="🚀")

st.title("📊 نظام التنبؤ باتجاه أسعار البيتكوين")
st.write("هذا النظام يستخدم خوارزمية Logistic Regression لتحليل البيانات المالية.")

# 2. تحميل المودل (الذي تدربنا عليه مسبقاً)
@st.cache_resource # لتسريع التطبيق وعدم تحميل المودل في كل مرة
def load_my_model():
    model = joblib.load('bitcoin_logistic_model.pkl')
    return model

model = load_my_model()

# 1. إضافة خانات الإدخال الأربع (مع إضافة سعر الأمس لحساب العائد)
st.sidebar.header("مدخلات البيانات اليومية")
close_price = st.sidebar.number_input("سعر الإغلاق الحالي ($)", value=60000.0)
prev_close = st.sidebar.number_input("سعر إغلاق الأمس ($)", value=59800.0)
ma_7 = st.sidebar.number_input("المتوسط المتحرك (Moving_Avarage_(7Days))", value=59500.0)
volatility = st.sidebar.number_input("معدل التذبذب (Volatility)", value=0.02)

# 2. حساب العائد (Return) برمجياً
# العائد هو نسبة التغير بين سعر اليوم وسعر الأمس
current_return = (close_price - prev_close) / prev_close

# زر التوقع
if st.button("توقع الاتجاه"):
    # 3. ترتيب البيانات تماماً كما في 'selected_features'
    # الترتيب: Close, MA_7, Return, Volatility
    features = np.array([[close_price, ma_7, current_return, volatility]])
    
    prediction = model.predict(features)
    # عرض النتيجة
    st.subheader("النتيجة التحليلية:")
    if prediction[0] == 1:
        st.success("🚀 التوقع: صعود (Price will Go Up)")
    else:
        st.error("📉 التوقع: هبوط (Price will Go Down)")

    # إضافة لمسة هندسية: عرض القيم التي تم إرسالها للمودل
    st.write("الميزات المرسلة للمودل:", features)


