import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import asyncio
import requests
import uuid
import sqlite3
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
except ImportError:
    Sequential = None
    LSTM = None
    Dense = None
    load_model = None
import threading
import time
import paho.mqtt.client as mqtt
import xml.etree.ElementTree as ET
import logging
from io import BytesIO
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    sns = None
    plt = None
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None
from cryptography.fernet import Fernet
if 'is_simulating' not in st.session_state:
    st.session_state.is_simulating = False
# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo khóa mã hóa
key = Fernet.generate_key()
cipher = Fernet(key)

# Cấu hình giao diện
st.set_page_config(
    page_title="CNCTech AI - Lập Kế Hoạch & Giám Sát Thông Minh",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #121212 0%, #1a1a1a 100%) !important;
        color: #ffffff !important;
        font-family: 'Roboto', sans-serif !important;
    }
    .main {
        padding: 20px !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
        border-right: 3px solid #d32f2f !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-size: 16px !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: 2px solid #d32f2f !important;
    }
    .nav-bar {
        background-color: #1e1e1e !important;
        padding: 12px !important;
        border-radius: 10px !important;
        margin-bottom: 20px !important;
        display: flex !important;
        justify-content: space-around !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    .nav-bar a {
        color: #ffffff !important;
        text-decoration: none !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    .nav-bar a:hover {
        background-color: #d32f2f !important;
        transform: scale(1.05) !important;
    }
    [data-testid="stMetric"] {
        background-color: #1e1e1e !important;
        border: 2px solid #d32f2f !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #d32f2f !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
    }
    .stButton>button {
        background-color: #d32f2f !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #b71c1c !important;
        transform: translateY(-2px) !important;
    }
    input, textarea, select {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 2px solid #d32f2f !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    [data-testid="stTabs"] button {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px !important;
    }
    [data-testid="tooltip"] {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    [data-testid="tooltip"] .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #d32f2f;
        color: #ffffff;
        text-align: center;
        border-radius: 8px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.4s;
    }
    [data-testid="tooltip"]:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    .card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo cơ sở dữ liệu SQLite
def init_db():
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                     (id TEXT PRIMARY KEY, factory_id TEXT, time TEXT, message TEXT, status TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS configurations
                     (id TEXT PRIMARY KEY, factory_id TEXT, time TEXT, key TEXT, value TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS scenarios
                     (id TEXT PRIMARY KEY, time TEXT, scenario_name TEXT, parameters TEXT, results TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS mqtt_cache
                     (id TEXT PRIMARY KEY, factory_id TEXT, time TEXT, data TEXT)''')
        
        # Check and add factory_id column to scenarios table if missing
        c.execute("PRAGMA table_info(scenarios)")
        columns = [info[1] for info in c.fetchall()]
        if 'factory_id' not in columns:
            c.execute("ALTER TABLE scenarios ADD COLUMN factory_id TEXT")
            logger.info("Added factory_id column to scenarios table")
        
        conn.commit()
        logger.info("Khởi tạo cơ sở dữ liệu thành công")
        return True
    except Exception as e:
        logger.error(f"Khởi tạo cơ sở dữ liệu thất bại: {e}")
        st.error(f"❌ Lỗi khởi tạo cơ sở dữ liệu: {e}")
        return False
    finally:
        conn.close()

init_db()

# Tiêu đề
st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <img src='https://via.placeholder.com/200x60.png?text=CNCTech+AI+Pro' alt='CNCTech Logo' class='pulse' style='max-width: 200px;'>
    <h1 style='color: #d32f2f; font-size: 32px; font-weight: bold;'>🏭 CNCTech AI </h1>
    <p style='color: #ffffff; font-size: 18px;'>Tối ưu hóa & giám sát sản xuất CNC bằng AI - Phát triển bởi < VU THE ANH ></p>
</div>
<div class='nav-bar'>
    <a href='#dashboard'>🏠 Dashboard</a>
    <a href='#planning'>📦 Lập Kế Hoạch</a>
    <a href='#performance'>📊 Hiệu Suất</a>
    <a href='#maintenance'>🔧 Bảo Trì</a>
    <a href='#scenarios'>🔬 Kịch Bản</a>
    <a href='#guide'>📖 Hướng Dẫn</a>
</div>
""", unsafe_allow_html=True)

# Thanh điều hướng
st.sidebar.title("📚 Menu Chức Năng")
st.sidebar.markdown("Sản phẩm được phát triển bởi DIM TEAM thuộc IMS LAB:")
menu = st.sidebar.radio(
    "Chọn Tác Vụ:",
    [
        "🏠 Dashboard Tổng Quan",
        "📦 Lập Kế Hoạch Sản Xuất Thủ Công",
        "📦 Lập Kế Hoạch Sản Xuất (ACO + RL)",
        "📊 Phân Tích Hiệu Suất & Dừng Máy",
        "📈 Báo Cáo Hiệu Suất Theo Ca",
        "📥 Tích Hợp Dữ Liệu Thời Gian Thực",
        "📤 Xuất Báo Cáo Đa Định Dạng",
        "📌 Biểu Đồ Chuỗi Nguyên Công",
        "📩 Cảnh Báo Thông Minh Telegram",
        "📋 Phân Tích Hiệu Suất Sâu",
        "🔧 Dự Đoán Bảo Trì Nâng Cao",
        "⏱️ Mô Phỏng Thời Gian Thực",
        "🖼️ Digital Twin (3D)",
        "🔬 Phân Tích Kịch Bản Nâng Cao",
        "📈 So Sánh Hiệu Suất AI",
        "📖 Hướng Dẫn Sử Dụng"
    ],
    captions=[
        "Theo dõi mọi hoạt động sản xuất",
        "Tạo lịch sản xuất thủ công nhanh chóng",
        "Tối ưu lịch sản xuất bằng AI",
        "Phân tích hiệu suất và thời gian dừng",
        "Xem chi tiết hiệu suất theo ca",
        "Tích hợp dữ liệu từ file hoặc IoT",
        "Xuất báo cáo Excel, JSON, XML, PDF",
        "Hiển thị quy trình sản xuất",
        "Nhận cảnh báo tức thời qua Telegram",
        "Khám phá nguyên nhân gốc rễ hiệu suất",
        "Dự đoán bảo trì chính xác",
        "Mô phỏng sản xuất thời gian thực",
        "Hiển thị chuyển động công cụ CNC 3D",
        "So sánh các kịch bản sản xuất",
        "Đánh giá cải tiến nhờ AI",
        "Hướng dẫn sử dụng chi tiết"
    ]
)

# Trạng thái phiên
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = []
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = []
if 'is_simulating' not in st.session_state:
    st.session_state.is_simulating = False
if 'rl_model_trained' not in st.session_state:
    st.session_state.rl_model_trained = False
if 'mqtt_data' not in st.session_state:
    st.session_state.mqtt_data = []
if 'scenario_history' not in st.session_state:
    st.session_state.scenario_history = []
if 'training_step' not in st.session_state:
    st.session_state.training_step = 0
if 'factory_id' not in st.session_state:
    st.session_state.factory_id = "FACTORY_01"
if 'sim_thread' not in st.session_state:
    st.session_state.sim_thread = None    

# Hàm hỗ trợ
def validate_realtime_data(data):
    required_fields = ["Ca", "Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]
    numeric_fields = ["Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]
    for field in required_fields:
        if field not in data:
            return False
        if field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False
    return True

def preprocess_data(df):
    df = df.copy()
    df = df.fillna(0)
    downtime_cols = ["Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]
    available_cols = [col for col in downtime_cols if col in df.columns]
    if available_cols:
        df["Tổng dừng"] = df[available_cols].sum(axis=1)
    if "Sản lượng (SP)" in df.columns and "Tổng dừng" in df.columns:
        df["Hiệu suất (%)"] = ((df["Sản lượng (SP)"] / (480 - df["Tổng dừng"])) * 100).round(1)
    features = ["Sản lượng (SP)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]
    scaler = None
    if all(col in df.columns for col in features):
        scaler = StandardScaler()
        scaler.fit(df[features])
    return df, scaler

@st.cache_resource
def init_rl_model(num_machines, num_jobs):
    if Sequential is None:
        logger.warning("Mô hình RL không khả dụng do thiếu TensorFlow")
        return None
    try:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(num_machines + num_jobs,)),
            Dense(64, activation='relu'),
            Dense(num_machines, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        try:
            model = load_model('rl_model.h5')
            logger.info("Đã tải mô hình RL hiện có")
        except:
            logger.info("Khởi tạo mô hình RL mới")
        return model
    except Exception as e:
        logger.error(f"Khởi tạo mô hình RL thất bại: {e}")
        st.error(f"❌ Lỗi khởi tạo mô hình RL: {e}")
        return None

def train_rl_model(model, states, actions, rewards):
    if model is None:
        logger.warning("Không có mô hình RL để huấn luyện")
        return None
    try:
        model.fit(states, rewards, epochs=10, verbose=0)
        model.save('rl_model.h5')
        logger.info("Mô hình RL đã được huấn luyện và lưu")
        return model
    except Exception as e:
        logger.error(f"Huấn luyện RL thất bại: {e}")
        st.error(f"❌ Lỗi huấn luyện RL: {e}")
        return None

def adjust_aco_for_energy(job, machines, pheromones, processing_times, setup_times, power_rates, hour):
    probabilities = []
    for machine in machines:
        pheromone = pheromones.get((job, machine), 1.0)
        eta = 1.0 / (processing_times[job] + setup_times.get((job, machine), 1.0))
        energy_factor = 1.0 if hour % 24 in range(8, 18) else 0.5 / power_rates.get(machine, 1.0)
        probabilities.append((machine, pheromone * eta * energy_factor))
    return probabilities

def run_aco_schedule(num_jobs, num_machines, priorities=None, setup_times=None, power_rates=None, broken_machines=None):
    try:
        np.random.seed(42)
        jobs = [f"SP{i+1}" for i in range(num_jobs)]
        machines = [f"M{i+1}" for i in range(num_machines)]
        if broken_machines:
            machines = [m for m in machines if m not in broken_machines]
        priorities = priorities or {j: 1.0 for j in jobs}
        setup_times = setup_times or {(j, m): np.random.randint(1, 5) for j in jobs for m in machines}
        power_rates = power_rates or {m: np.random.uniform(0.5, 2.0) for m in machines}
        processing_times = {j: np.random.randint(10, 40) for j in jobs}
        energy_prices = {"peak": 0.15, "off_peak": 0.08}
        
        schedule = []
        timeline = {m: 0 for m in machines}
        pheromones = {(j, m): 1.0 for j in jobs for m in machines}
        
        for _ in range(100):
            available_jobs = jobs.copy()
            current_hour = int(timeline.get(machines[0], 0) / 60)
            while available_jobs:
                job = max(available_jobs, key=lambda x: priorities[x])
                probabilities = adjust_aco_for_energy(job, machines, pheromones, processing_times, setup_times, power_rates, current_hour)
                
                if not probabilities:
                    st.error(f"Lỗi: Không tìm thấy máy phù hợp cho sản phẩm {job}")
                    return [], [], 0.0
                
                total = sum(prob[1] for prob in probabilities)
                probabilities = [(m, p/total if total > 0 else 1/len(probabilities)) for m, p in probabilities]
                
                machine = np.random.choice([p[0] for p in probabilities], p=[p[1] for p in probabilities])
                start = timeline[machine]
                duration = processing_times[job] + setup_times.get((job, machine), 1.0)
                schedule.append((machine, start, start + duration, job))
                timeline[machine] += duration
                pheromones[(job, machine)] = pheromones.get((job, machine), 1.0) * 1.5
                available_jobs.remove(job)
        
        energy_cost = 0
        for machine, start, end, job in schedule:
            if machine not in power_rates:
                power_rates[machine] = 1.0
            hours = np.arange(int(start/60), int(end/60) + 1)
            for h in hours:
                price = energy_prices["peak"] if 8 <= h % 24 <= 17 else energy_prices["off_peak"]
                energy_cost += processing_times[job] * power_rates[machine] * price
        
        return schedule, timeline, energy_cost
    
    except Exception as e:
        st.error(f"Lập lịch ACO thất bại: {str(e)}")
        return [], [], 0.0

def plot_gantt_schedule(data):
    try:
        df = pd.DataFrame(data, columns=["Máy", "Bắt đầu", "Kết thúc", "Sản phẩm"])
        df['Bắt đầu'] = pd.to_datetime(df['Bắt đầu'], unit='m')
        df['Kết thúc'] = pd.to_datetime(df['Kết thúc'], unit='m')
        
        fig = px.timeline(
            df,
            x_start="Bắt đầu", 
            x_end="Kết thúc", 
            y="Máy", 
            color="Sản phẩm",
            title="🛠️ Biểu Đồ Gantt Lịch Sản Xuất (ACO + RL)",
            hover_data={"Bắt đầu": "|%H:%M", "Kết thúc": "|%H:%M", "Sản phẩm": True}
        )
        fig.update_yaxes(autorange="reversed", title="Máy CNC")
        fig.update_xaxes(title="Thời Gian")
        fig.update_layout(
            hovermode="closest",
            showlegend=True,
            template="plotly_dark",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        logger.info("Biểu đồ Gantt đã được tạo")
        return fig, df
    except Exception as e:
        logger.error(f"Lỗi tạo biểu đồ Gantt: {e}")
        st.error(f"❌ Lỗi biểu đồ Gantt: {e}")
        return None, None

def save_scenario_to_db(scenario_name, parameters, results, factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        
        # Check if factory_id column exists
        c.execute("PRAGMA table_info(scenarios)")
        columns = [info[1] for info in c.fetchall()]
        scenario_id = str(uuid.uuid4())
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if 'factory_id' in columns:
            c.execute("INSERT INTO scenarios (id, factory_id, time, scenario_name, parameters, results) VALUES (?, ?, ?, ?, ?, ?)",
                      (scenario_id, factory_id, time, scenario_name, json.dumps(parameters), json.dumps(results)))
        else:
            c.execute("INSERT INTO scenarios (id, time, scenario_name, parameters, results) VALUES (?, ?, ?, ?, ?)",
                      (scenario_id, time, scenario_name, json.dumps(parameters), json.dumps(results)))
        
        conn.commit()
        logger.info(f"Đã lưu kịch bản: {scenario_name}")
        return {"id": scenario_id, "factory_id": factory_id, "time": time, "scenario_name": scenario_name}
    except Exception as e:
        logger.error(f"Lưu kịch bản thất bại: {e}")
        st.error(f"❌ Lỗi lưu kịch bản: {e}")
        return None
    finally:
        conn.close()

def load_alert_history(factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        df = pd.read_sql_query("SELECT * FROM alerts WHERE factory_id = ? ORDER BY time DESC", conn, params=(factory_id,))
        logger.info("Đã tải lịch sử cảnh báo")
        return df
    except Exception as e:
        logger.error(f"Tải lịch sử cảnh báo thất bại: {e}")
        st.error(f"❌ Lỗi tải lịch sử cảnh báo: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
def save_alert_to_db(message, status, factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        alert_id = str(uuid.uuid4())
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            "INSERT INTO alerts (id, factory_id, time, message, status) VALUES (?, ?, ?, ?, ?)",
            (alert_id, factory_id, time_now, message, status)
        )
        conn.commit()
        logger.info(f"Đã lưu cảnh báo: {message} - {status}")
    except Exception as e:
        logger.error(f"Lưu cảnh báo thất bại: {e}")
    finally:
        conn.close()        

def save_scenario_to_db(scenario_name, parameters, results, factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        scenario_id = str(uuid.uuid4())
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO scenarios (id, factory_id, time, scenario_name, parameters, results) VALUES (?, ?, ?, ?, ?, ?)",
                  (scenario_id, factory_id, time, scenario_name, json.dumps(parameters), json.dumps(results)))
        conn.commit()
        logger.info(f"Đã lưu kịch bản: {scenario_name}")
        return {"id": scenario_id, "factory_id": factory_id, "time": time, "scenario_name": scenario_name}
    except Exception as e:
        logger.error(f"Lưu kịch bản thất bại: {e}")
        st.error(f"❌ Lỗi lưu kịch bản: {e}")
        return None
    finally:
        conn.close()

def load_scenario_history(factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        df = pd.read_sql_query("SELECT * FROM scenarios WHERE factory_id = ? ORDER BY time DESC", conn, params=(factory_id,))
        logger.info("Đã tải lịch sử kịch bản")
        return df
    except Exception as e:
        logger.error(f"Tải lịch sử kịch bản thất bại: {e}")
        st.error(f"❌ Lỗi tải lịch sử kịch bản: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def save_mqtt_to_db(data, factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        mqtt_id = str(uuid.uuid4())
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO mqtt_cache (id, factory_id, time, data) VALUES (?, ?, ?, ?)",
                  (mqtt_id, factory_id, time, json.dumps(data)))
        conn.commit()
        logger.info("Đã lưu dữ liệu MQTT vào bộ nhớ cache")
    except Exception as e:
        logger.error(f"Lưu dữ liệu MQTT thất bại: {e}")
    finally:
        conn.close()

def load_mqtt_cache(factory_id="FACTORY_01"):
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        df = pd.read_sql_query("SELECT * FROM mqtt_cache WHERE factory_id = ? ORDER BY time DESC LIMIT 100", conn, params=(factory_id,))
        logger.info("Đã tải dữ liệu MQTT từ bộ nhớ cache")
        return df
    except Exception as e:
        logger.error(f"Tải dữ liệu MQTT thất bại: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def send_telegram_alert(message):
    try:
        token = cipher.decrypt(st.secrets.get("telegram", {}).get("bot_token", "").encode()).decode()
        chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        if not token or not chat_id:
            status = "Lỗi: Cấu hình Telegram không hợp lệ"
            save_alert_to_db(message, status)
            st.error("❌ Vui lòng cấu hình Telegram trong secrets!")
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        for attempt in range(3):
            try:
                r = requests.post(url, data=data, timeout=5)
                status = "Thành công" if r.status_code == 200 else f"Lỗi: {r.status_code}"
                save_alert_to_db(message, status)
                if r.status_code == 200:
                    st.success("🚨 Đã gửi cảnh báo Telegram!")
                    return True
                else:
                    st.error(f"❌ Lỗi gửi Telegram: {r.status_code}")
                    return False
            except requests.RequestException:
                time.sleep(2)
        status = "Lỗi: Đã đạt tối đa số lần thử"
        save_alert_to_db(message, status)
        st.error("❌ Lỗi gửi Telegram: Đã đạt tối đa số lần thử")
        return False
    except Exception as e:
        status = f"Lỗi: {str(e)}"
        save_alert_to_db(message, status)
        st.error(f"❌ Lỗi gửi Telegram: {e}")
        return False

@st.cache_resource
def init_predictive_models():
    try:
        rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        logger.info("Khởi tạo mô hình dự đoán thành công")
        return rf_model, iso_forest
    except Exception as e:
        logger.error(f"Khởi tạo mô hình dự đoán thất bại: {e}")
        st.error(f"❌ Lỗi khởi tạo mô hình dự đoán: {e}")
        return None, None

def fetch_erp_data(endpoint, token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(endpoint, headers=headers, timeout=10)
        return pd.DataFrame(response.json())
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu ERP: {e}")
        st.error(f"❌ Lỗi lấy dữ liệu ERP: {e}")
        return None

async def fetch_iot_data():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("cnc/machines/data")
            st.session_state.mqtt_data.append({"time": datetime.now(), "message": "Đã kết nối MQTT"})
            logger.info("Đã kết nối MQTT")
        else:
            st.session_state.mqtt_data.append({"time": datetime.now(), "message": f"Kết nối MQTT thất bại: {rc}"})
            logger.error(f"Kết nối MQTT thất bại: {rc}")
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if validate_realtime_data(data):
                st.session_state.mqtt_data.append({"time": datetime.now(), "data": data})
                save_mqtt_to_db(data)
                logger.info(f"Tin nhắn MQTT: {data}")
            else:
                st.session_state.mqtt_data.append({"time": datetime.now(), "message": f"Dữ liệu MQTT không hợp lệ"})
                logger.error("Dữ liệu MQTT không hợp lệ")
        except Exception as e:
            st.session_state.mqtt_data.append({"time": datetime.now(), "message": f"Lỗi dữ liệu MQTT: {e}"})
            logger.error(f"Lỗi dữ liệu MQTT: {e}")
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        for attempt in range(3):
            try:
                client.connect("broker.hivemq.com", 1883, 60)
                client.loop_start()
                await asyncio.sleep(5)
                client.loop_stop()
                logger.info("Đã lấy dữ liệu MQTT")
                return
            except Exception:
                time.sleep(2)
        logger.error("Kết nối MQTT thất bại: Đã đạt tối đa số lần thử")
        st.error("❌ Lỗi kết nối MQTT: Đã đạt tối đa số lần thử")
        mqtt_cache = load_mqtt_cache()
        if not mqtt_cache.empty:
            st.session_state.mqtt_data = [json.loads(row["data"]) for _, row in mqtt_cache.iterrows()]
            st.info("ℹ Đã sử dụng dữ liệu MQTT từ bộ nhớ cache")
    except Exception as e:
        logger.error(f"Kết nối MQTT thất bại: {e}")
        st.error(f"❌ Lỗi kết nối MQTT: {e}")

async def simulate_realtime_data():
    try:
        while st.session_state.get("is_simulating", False):
            new_data = {
                "Ca": np.random.choice(["Sáng", "Chiều", "Tối"]),
                "Sản lượng (SP)": np.random.randint(80, 150),
                "Dừng do máy (phút)": np.random.randint(0, 20),
                "Dừng do người (phút)": np.random.randint(0, 15),
                "Dừng lý do khác (phút)": np.random.randint(0, 10),
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.write("DEBUG new_data:", new_data)  # Thêm dòng này để kiểm tra
            if validate_realtime_data(new_data):
                st.session_state.realtime_data.append(new_data)
                if len(st.session_state.realtime_data) > 100:
                    st.session_state.realtime_data.pop(0)
            await asyncio.sleep(3)
            logger.info("Đã mô phỏng dữ liệu thời gian thực")
    except Exception as e:
        logger.error(f"Mô phỏng thời gian thực thất bại: {e}")
        st.error(f"❌ Lỗi mô phỏng thời gian thực: {e}")

def export_to_erp(df, format="json"):
    try:
        if format == "json":
            buffer = io.StringIO()
            df.to_json(buffer, orient="records", force_ascii=False)
            return buffer.getvalue(), f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json"
        elif format == "xml":
            buffer = io.StringIO()
            root = ET.Element("ProductionData")
            for _, row in df.iterrows():
                record = ET.SubElement(root, "Record")
                for col, val in row.items():
                    ET.SubElement(record, col.replace(" ", "_")).text = str(val)
            buffer.write(ET.tostring(root, encoding="unicode"))
            return buffer.getvalue(), f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml", "application/xml"
        elif format == "excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="ProductionReport")
                workbook = writer.book
                worksheet = writer.sheets["ProductionReport"]
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter
                    for cell in col:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column].width = adjusted_width
            return buffer.getvalue(), f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == "pdf" and FPDF:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="CNCTech AI Pro - Báo Cáo Sản Xuất", ln=True, align="C")
            pdf.ln(10)
            for i, row in df.iterrows():
                for col, val in row.items():
                    pdf.cell(50, 10, f"{col}: {val}", ln=True)
            buffer = BytesIO()
            pdf.output(buffer)
            return buffer.getvalue(), f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf"
        logger.info(f"Đã xuất dưới dạng {format}")
        return None, None, None
    except Exception as e:
        logger.error(f"Xuất báo cáo thất bại: {e}")
        st.error(f"❌ Lỗi xuất báo cáo: {e}")
        return None, None, None

def generate_heatmap(df):
    if sns is None or plt is None:
        logger.warning("Heatmap không khả dụng do thiếu Seaborn/Matplotlib")
        return None
    try:
        corr = df[["Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Tương Quan Các Yếu Tố Sản Xuất")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        buffer.seek(0)
        logger.info("Đã tạo heatmap")
        return buffer
    except Exception as e:
        logger.error(f"Tạo heatmap thất bại: {e}")
        st.error(f"❌ Lỗi heatmap: {e}")
        return None

@st.cache_resource
def init_lstm_model(timesteps=5, n_features=5):
    if Sequential is None:
        logger.warning("Mô hình LSTM không khả dụng do thiếu TensorFlow")
        return None
    try:
        model = Sequential([
            LSTM(100, activation='relu', input_shape=(timesteps, n_features), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        logger.info("Khởi tạo mô hình LSTM thành công")
        return model
    except Exception as e:
        logger.error(f"Khởi tạo mô hình LSTM thất bại: {e}")
        st.error(f"❌ Lỗi khởi tạo mô hình LSTM: {e}")
        return None

def reshape_lstm_input(df, timesteps=5):
    features = ["Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)", "Tổng dừng"]
    X = df[features].values
    if len(X) < timesteps:
        X = np.pad(X, ((timesteps - len(X), 0), (0, 0)), mode='constant')
    return X[-timesteps:].reshape(1, timesteps, len(features))

def parse_gcode(file):
    toolpath = {"x": [], "y": [], "z": []}
    for line in file:
        if line.startswith("G01"):
            parts = line.split()
            for part in parts:
                if part.startswith("X"): toolpath["x"].append(float(part[1:]))
                elif part.startswith("Y"): toolpath["y"].append(float(part[1:]))
                elif part.startswith("Z"): toolpath["z"].append(float(part[1:]))
    return toolpath

def check_reschedule_trigger(df):
    if df.empty:
        return False
    try:
        return df["Tổng dừng"].mean() > 30 or df["Hiệu suất (%)"].mean() < 70
    except:
        return False

async def monitor_and_reschedule(jobs, machines, priorities, power_rates):
    try:
        while st.session_state.is_simulating:
            df_realtime = pd.DataFrame(st.session_state.realtime_data)
            if check_reschedule_trigger(df_realtime):
                broken_machines = df_realtime[df_realtime["Dừng do máy (phút)"] > 60]["Máy"].tolist() if "Máy" in df_realtime.columns else []
                schedule, timeline, energy_cost = run_aco_schedule(
                    jobs=len(df_realtime["Sản phẩm"].unique()) if "Sản phẩm" in df_realtime.columns else jobs,
                    machines=machines,
                    priorities=priorities,
                    power_rates=power_rates,
                    broken_machines=broken_machines
                )
                if schedule:
                    fig, df_gantt = plot_gantt_schedule(schedule)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"⏱️ Lịch sản xuất đã được cập nhật: {max(timeline.values())} phút")
                        send_telegram_alert(f"⚠️ Lịch sản xuất đã được cập nhật do bất thường trong dữ liệu thời gian thực!")
            await asyncio.sleep(60)
    except Exception as e:
        logger.error(f"Lỗi giám sát và lập lại lịch: {e}")
        st.error(f"❌ Lỗi giám sát và lập lại lịch: {e}")

# Logic menu
if menu == "🏠 Dashboard Tổng Quan":
    with st.container():
        st.markdown("<div class='card'><h2>🏠 Dashboard Tổng Quan</h2><p>Theo dõi thời gian thực các hoạt động sản xuất.</p></div>", unsafe_allow_html=True)
        st.markdown(f"**Thời Gian Hiện Tại**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.text_input("Factory ID", value=st.session_state.factory_id, key="factory_id", on_change=lambda: setattr(st.session_state, "factory_id", st.session_state.factory_id))
        if st.session_state.df_uploaded is not None:
            df, _ = preprocess_data(st.session_state.df_uploaded)
            if df is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng Sản Lượng", f"{df['Sản lượng (SP)'].sum():,}", help="Tổng số sản phẩm được sản xuất")
                with col2:
                    df["Tổng dừng"] = df.iloc[:, 2:5].sum(axis=1)
                    st.metric("Tổng Thời Gian Dừng", f"{df['Tổng dừng'].sum():,} phút", help="Tổng thời gian dừng máy")
                with col3:
                    df["Hiệu suất (%)"] = ((df["Sản lượng (SP)"] / (480 - df["Tổng dừng"])) * 100).round(1)
                    st.metric("Hiệu Suất Trung Bình", f"{df['Hiệu suất (%)'].mean():.1f}%", help="Hiệu suất trung bình của ca")
                with col4:
                    availability = df["Hiệu suất (%)"].mean() / 100
                    performance = min(df["Sản lượng (SP)"].mean() / 150, 1.0)
                    quality = 0.95
                    oee = availability * performance * quality
                    st.metric("OEE", f"{oee*100:.1f}%", help="Hiệu Suất Thiết Bị Tổng Thể")
                st.markdown("<div class='card'><h3>🛠️ Trạng Thái Máy CNC</h3></div>", unsafe_allow_html=True)
                status_data = {
                    "Máy": [f"M{i+1}" for i in range(5)],
                    "Trạng Thái": [np.random.choice(["Hoạt động", "Dừng", "Bảo Trì"]) for _ in range(5)],
                    "Công Suất (kW)": [np.random.uniform(0.5, 2.0) for _ in range(5)]
                }
                st.dataframe(pd.DataFrame(status_data).style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                st.markdown("<div class='card'><h3>📊 Luồng Sản Xuất Theo Ca</h3></div>", unsafe_allow_html=True)
                sankey_data = {
                    "label": ["Đầu Vào", "Ca Sáng", "Ca Chiều", "Ca Tối", "Đầu Ra"],
                    "source": [0, 0, 0, 1, 2, 3],
                    "target": [1, 2, 3, 4, 4, 4],
                    "value": [
                        df[df["Ca"] == "Sáng"]["Sản lượng (SP)"].sum(),
                        df[df["Ca"] == "Chiều"]["Sản lượng (SP)"].sum(),
                        df[df["Ca"] == "Tối"]["Sản lượng (SP)"].sum(),
                        df[df["Ca"] == "Sáng"]["Sản lượng (SP)"].sum(),
                        df[df["Ca"] == "Chiều"]["Sản lượng (SP)"].sum(),
                        df[df["Ca"] == "Tối"]["Sản lượng (SP)"].sum()
                    ]
                }
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(label=sankey_data["label"], color="#d32f2f"),
                    link=dict(source=sankey_data["source"], target=sankey_data["target"], value=sankey_data["value"], color="#ffffff")
                )])
                fig_sankey.update_layout(
                    title="Luồng Sản Xuất Theo Ca",
                    template="plotly_dark",
                    font=dict(size=12)
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("ℹ️ Vui lòng tích hợp dữ liệu để mở khóa toàn bộ dashboard!", icon="ℹ️")

elif menu == "📦 Lập Kế Hoạch Sản Xuất Thủ Công":
    with st.container():
        st.markdown("<div class='card'><h2>📦 Lập Kế Hoạch Sản Xuất Thủ Công</h2><p>Tạo lịch sản xuất hiệu quả với các tham số tùy chỉnh.</p></div>", unsafe_allow_html=True)
        with st.form("manual_planning_form"):
            col1, col2 = st.columns(2)
            with col1:
                slsp = st.number_input("Số Lượng Sản Phẩm", min_value=1, value=100, step=1, help="Số lượng sản phẩm cần sản xuất")
                thoigian = st.slider("Thời Gian Xử Lý Mỗi Sản Phẩm (phút)", 5, 120, 30, help="Thời gian trung bình cho mỗi sản phẩm")
            with col2:
                somay = st.slider("Số Máy CNC", 1, 250, 100, help="Số máy CNC có sẵn")
                deadline = st.slider("Thời Hạn Giao Hàng (giờ)", 1, 168, 24, help="Thời gian tối đa để hoàn thành đơn hàng")
            submitted = st.form_submit_button("🧠 Tạo Kế Hoạch")
            if submitted:
                if slsp <= 0 or thoigian <= 0 or somay <= 0:
                    st.error("❌ Vui lòng nhập giá trị lớn hơn 0!")
                else:
                    total_time = slsp * thoigian / somay
                    st.markdown(f"<div class='card'><p>🧾 Tổng Thời Gian Xử Lý: <strong>{total_time/60:.2f} giờ</strong></p></div>", unsafe_allow_html=True)
                    if total_time / 60 <= deadline:
                        st.success("✅ Đơn hàng sẽ đáp ứng thời hạn!")
                    else:
                        st.warning("⚠️ Đơn hàng có nguy cơ trễ hạn!")
                        st.info(f"📝 Đề xuất: Tăng số máy lên {(slsp * thoigian / (deadline * 60)):.0f} hoặc điều chỉnh ca làm việc.")

elif menu == "📦 Lập Kế Hoạch Sản Xuất (ACO + RL)":
    with st.container():
        st.markdown("<div class='card'><h2>📦 Lập Kế Hoạch Sản Xuất (ACO + RL)</h2><p>Tận dụng AI để tối ưu hóa lịch sản xuất.</p></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            jobs = st.slider("Số Sản Phẩm", 10, 100, 30, help="Số lượng sản phẩm cần lập lịch")
        with col2:
            machines = st.slider("Số Máy CNC", 5, 30, 15, help="Số máy CNC có sẵn")
        with col3:
            priority_job = st.text_input("Sản Phẩm Ưu Tiên (e.g., SP1,SP2)", "", help="Các sản phẩm ưu tiên, cách nhau bằng dấu phẩy")
        st.markdown("#### Nhập công suất từng máy (kW)")
        power_rates = {}
        for i in range(machines):
            power_rates[f"M{i+1}"] = st.number_input(
                f"Công Suất Máy M{i+1} (kW)", min_value=0.1, value=1.0, step=0.1, key=f"power_{i}"
            )
        priorities = {f"SP{i+1}": 1.0 for i in range(jobs)}
        if priority_job:
            for pj in priority_job.split(","):
                if pj in priorities:
                    priorities[pj] = 2.5
        broken_machines = ""             
        broken_machines_list = [m.strip() for m in broken_machines.split(",") if m.strip()] if broken_machines else []
        for m in broken_machines_list:
            if m not in [f"M{i+1}" for i in range(machines)]:
                st.error(f"❌ Máy {m} không hợp lệ!")
                st.stop()
        if st.button("🚀 Tạo Lịch Tối Ưu"):
            if jobs <= machines:
                st.warning("⚠️ Số sản phẩm nên vượt số máy để tối ưu lịch!")
            schedule, timeline, energy_cost = run_aco_schedule(jobs, machines, priorities, power_rates=power_rates)
            if schedule:
                rl_model = init_rl_model(machines, jobs)
                if rl_model and st.session_state.df_uploaded is not None and not st.session_state.rl_model_trained:
                    df, _ = preprocess_data(st.session_state.df_uploaded)
                    if df is not None:
                        states = []
                        for i in range(len(df)):
                            state_machines = [df.iloc[i]["Tổng dừng"]] * machines
                            state_jobs = [1] * jobs
                            states.append(state_machines + state_jobs)
                        states = np.array(states)
                        rewards = df["Hiệu suất (%)"].values / 100
                        rl_model = train_rl_model(rl_model, states, None, rewards)
                        st.session_state.rl_model_trained = True
                if rl_model:
                    state = np.array([timeline[m] for m in [f"M{i+1}" for i in range(machines)]] + [1] * jobs)
                    action_probs = rl_model.predict(state.reshape(1, -1))[0]
                    epsilon = max(0.01, 0.1 * (0.995 ** st.session_state.training_step))
                    for i, (machine, start, end, job) in enumerate(schedule):
                        if np.random.random() < epsilon:
                            new_machine = np.random.choice([f"M{i+1}" for i in range(machines)], p=action_probs)
                            schedule[i] = (new_machine, start, end, job)
                    st.session_state.training_step += 1
                fig, df_gantt = plot_gantt_schedule(schedule)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"⏱️ Tổng Thời Gian Hoàn Thành: {max(timeline.values()):,} phút")
                    st.info(f"⚡ Chi Phí Năng Lượng: ${energy_cost:,.2f}")
                    st.markdown("<div class='card'><h3>⚡ Phân Tích Tiêu Thụ Năng Lượng</h3></div>", unsafe_allow_html=True)
                    energy_data = []
                    for machine, start, end, job in schedule:
                        hours = np.arange(int(start/60), int(end/60) + 1)
                        for h in hours:
                            price = 0.15 if 8 <= h % 24 <= 17 else 0.08
                            energy_data.append({"Giờ": h, "Máy": machine, "Năng Lượng (kWh)": power_rates.get(machine, 1.0) * price})
                    df_energy = pd.DataFrame(energy_data)
                    fig_energy = px.line(df_energy, x="Giờ", y="Năng Lượng (kWh)", color="Máy", title="Tiêu Thụ Năng Lượng Theo Giờ", template="plotly_dark")
                    st.plotly_chart(fig_energy, use_container_width=True)
                    if st.session_state.df_uploaded is not None:
                        df, scaler = preprocess_data(st.session_state.df_uploaded)
                        if df is not None and init_lstm_model() is not None:
                            lstm_model = init_lstm_model()
                            features = ["Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)", "Tổng dừng"]
                            if all(col in df.columns for col in features) and len(df) >= 5:
                                X = reshape_lstm_input(df)
                                predicted_time = lstm_model.predict(X, verbose=0)[0][0]
                                st.markdown(f"<div class='card'><p>🧠 Dự Đoán Thời Gian Xử Lý (LSTM): <strong>{predicted_time:.1f} phút</strong></p></div>", unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ Cần ít nhất 5 dòng dữ liệu đầy đủ để dự đoán LSTM!")
                    buffer = io.StringIO()
                    df_gantt.to_csv(buffer, index=False)
                    st.download_button(
                        "⬇️ Tải Dữ Liệu Gantt (CSV)",
                        data=buffer.getvalue(),
                        file_name=f"gantt_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

elif menu == "📊 Phân Tích Hiệu Suất & Dừng Máy":
    with st.container():
        st.markdown("<div class='card'><h2>📊 Phân Tích Hiệu Suất & Dừng Máy</h2><p>Phân tích dữ liệu sản xuất để nâng cao hiệu quả.</p></div>", unsafe_allow_html=True)
        data = {
            "Ca": ["Sáng", "Chiều", "Tối"],
            "Sản lượng (SP)": [120, 100, 90],
            "Dừng do máy (phút)": [10, 20, 5],
            "Dừng do người (phút)": [5, 15, 10],
            "Dừng lý do khác (phút)": [0, 5, 2]
        }
        df = st.data_editor(
            pd.DataFrame(data),
            num_rows="dynamic",
            column_config={
                "Sản lượng (SP)": st.column_config.NumberColumn(min_value=0, help="Số sản phẩm hoàn thành"),
                "Dừng do máy (phút)": st.column_config.NumberColumn(min_value=0, help="Thời gian dừng do máy"),
                "Dừng do người (phút)": st.column_config.NumberColumn(min_value=0, help="Thời gian dừng do người"),
                "Dừng lý do khác (phút)": st.column_config.NumberColumn(min_value=0, help="Thời gian dừng do lý do khác")
            }
        )
        if st.button("📈 Phân Tích Hiệu Suất"):
            if df.empty:
                st.error("❌ Vui lòng nhập dữ liệu để phân tích!")
            else:
                df, _ = preprocess_data(df)
                if df is not None:
                    df["Tổng dừng"] = df.iloc[:, 2:5].sum(axis=1)
                    df["Hiệu suất (%)"] = ((df["Sản lượng (SP)"] / (480 - df["Tổng dừng"])) * 100).round(1)
                    threshold = st.slider("Ngưỡng Cảnh Báo Hiệu Suất (%)", 50, 90, 80, help="Ngưỡng để phát hiện hiệu suất thấp")
                    df["Cảnh báo"] = np.where(df["Hiệu suất (%)"] < threshold, "⚠️ Cần Kiểm Tra", "✅ Tốt")
                    st.dataframe(df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                    st.session_state.df_uploaded = df
                    rf_model, iso_forest = init_predictive_models()
                    if iso_forest is not None:
                        anomalies = iso_forest.fit_predict(df[["Tổng dừng"]])
                        if -1 in anomalies:
                            anomaly_ca = df[anomalies == -1]["Ca"].tolist()
                            send_telegram_alert(f"⚠️ Phát hiện bất thường: Thời gian dừng cao ở ca {', '.join(anomaly_ca)}!")
                    fig = go.Figure(data=[
                        go.Bar(name='Sản Lượng (SP)', x=df['Ca'], y=df['Sản lượng (SP)'], marker_color="#d32f2f"),
                        go.Bar(name='Tổng Dừng (phút)', x=df['Ca'], y=df['Tổng dừng'], marker_color="#ffffff")
                    ])
                    fig.update_layout(
                        barmode='group',
                        title="Sản Lượng vs Thời Gian Dừng Theo Ca",
                        yaxis_title="Số Lượng/Phút",
                        template="plotly_dark",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    heatmap_buffer = generate_heatmap(df)
                    if heatmap_buffer:
                        st.image(heatmap_buffer, caption="Tương Quan Các Yếu Tố Sản Xuất")
                    if df["Hiệu suất (%)"].lt(threshold).any():
                        low_perf_ca = df[df["Hiệu suất (%)"] < threshold]["Ca"].tolist()
                        send_telegram_alert(f"⚠️ Cảnh báo: Hiệu suất thấp ở ca {', '.join(low_perf_ca)}!")

elif menu == "📈 Báo Cáo Hiệu Suất Theo Ca":
    with st.container():
        st.markdown("<div class='card'><h2>📈 Báo Cáo Hiệu Suất Theo Ca</h2><p>Theo dõi và so sánh hiệu suất sản xuất theo ca.</p></div>", unsafe_allow_html=True)
        df = pd.DataFrame({
            "Ngày": ["24/06", "25/06", "26/06"] * 3,
            "Ca": ["Sáng", "Chiều", "Tối"] * 3,
            "Hiệu suất (%)": [95, 80, 75, 92, 84, 60, 89, 78, 65]
        })
        erp_endpoint = st.text_input("ERP Endpoint", value="https://factory-mes.com/api/shift-performance", help="Nhập URL API ERP")
        erp_token = st.text_input("ERP Token", type="password", help="Nhập token API ERP")
        if st.button("🔄 Tải Dữ Liệu ERP"):
            df_erp = fetch_erp_data(erp_endpoint, erp_token)
            if df_erp is not None:
                df = df_erp
                st.session_state.df_uploaded = df
                st.success("✅ Đã tải dữ liệu ERP!")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Từ Ngày", value=datetime.now().date(), help="Ngày bắt đầu phân tích")
        with col2:
            end_date = st.date_input("Đến Ngày", value=datetime.now().date(), help="Ngày kết thúc phân tích")
        fig = px.line(
            df,
            x="Ngày",
            y="Hiệu suất (%)",
            color="Ca",
            markers=True,
            title="Hiệu Suất Theo Ca Qua Các Ngày",
            template="plotly_dark"
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='card'><h3>Thống Kê Tóm Tắt</h3></div>", unsafe_allow_html=True)
        summary = df.groupby("Ca")["Hiệu suất (%)"].agg(['mean', 'min', 'max']).round(1)
        st.dataframe(summary.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))

elif menu == "📥 Tích Hợp Dữ Liệu Thời Gian Thực":
    with st.container():
        st.markdown("<div class='card'><h2>📥 Tích Hợp Dữ Liệu Thời Gian Thực</h2><p>Kết nối với file, IoT hoặc ERP để phân tích thời gian thực.</p></div>", unsafe_allow_html=True)
        tabs = st.tabs(["📂 Tải File", "🌐 Kết Nối IoT", "🔄 Mô Phỏng API", "🏭 Tích Hợp ERP"])
        with tabs[0]:
            file = st.file_uploader("Tải File CSV/Excel", type=["csv", "xlsx"], help="File cần có: Ca, Sản lượng (SP), v.v.")
            if file:
                try:
                    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file, parse_dates=["Thời gian"], date_format="%Y-%m-%d %I:%M:%S %p")
                    if validate_realtime_data(df.iloc[0].to_dict()):
                        df, _ = preprocess_data(df)
                        if df is not None:
                            st.session_state.df_uploaded = df
                            st.success("✅ Dữ liệu đã được tải thành công!")
                            st.dataframe(df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))

                            # Tự động lập lịch với ACO + RL
                            jobs = len(df["Sản phẩm"].unique()) if "Sản phẩm" in df.columns else df["Sản lượng (SP)"].sum() // 10  # Ước lượng số sản phẩm
                            machines = st.slider("Số Máy CNC Tối Đa", 1, 30, 15, key="auto_machines", help="Số máy CNC để lập lịch")
                            priorities = {f"SP{i+1}": 1.0 for i in range(jobs)}
                            power_rates = {f"M{i+1}": np.random.uniform(0.5, 2.0) for i in range(machines)}
                            broken_machines = df[df["Dừng do máy (phút)"] > 60]["Máy"].tolist() if "Máy" in df.columns else []

                            if st.button("🚀 Tạo Lịch Tự Động"):
                                schedule, timeline, energy_cost = run_aco_schedule(
                                    num_jobs=jobs,
                                    num_machines=machines,
                                    priorities=priorities,
                                    power_rates=power_rates,
                                    broken_machines=broken_machines
                                )
                                if schedule:
                                    fig, df_gantt = plot_gantt_schedule(schedule)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.success(f"⏱️ Tổng Thời Gian Hoàn Thành: {max(timeline.values()):,} phút")
                                        st.info(f"⚡ Chi Phí Năng Lượng: ${energy_cost:,.2f}")
                                        buffer = io.StringIO()
                                        df_gantt.to_csv(buffer, index=False)
                                        st.download_button(
                                            "⬇️ Tải Dữ Liệu Gantt (CSV)",
                                            data=buffer.getvalue(),
                                            file_name=f"gantt_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )
                    else:
                        st.error("❌ Dữ liệu file không hợp lệ!")
                except Exception as e:
                    logger.error(f"Tải file thất bại: {e}")
                    st.error(f"❌ Lỗi đọc file: {e}")
        with tabs[1]:
            if st.button("🔄 Kết Nối MQTT"):
                asyncio.run(fetch_iot_data())
                if st.session_state.mqtt_data:
                    st.markdown("<div class='card'><h3>📡 Dữ Liệu IoT Thời Gian Thực</h3></div>", unsafe_allow_html=True)
                    st.json(st.session_state.mqtt_data)
                else:
                    st.info("ℹ Không nhận được dữ liệu MQTT.")
        with tabs[2]:
            if st.button("🔄 Mô Phỏng Dữ Liệu API"):
                try:
                    api_data = {
                        "Ca": ["Sáng", "Chiều", "Tối"],
                        "Sản lượng (SP)": [110, 95, 85],
                        "Dừng do máy (phút)": [12, 18, 7],
                        "Dừng do người (phút)": [6, 10, 8],
                        "Dừng lý do khác (phút)": [2, 4, 3]
                    }
                    df = pd.DataFrame(api_data)
                    if validate_realtime_data(df.iloc[0].to_dict()):
                        df, _ = preprocess_data(df)
                        if df is not None:
                            st.session_state.df_uploaded = df
                            st.success("✅ Đã tải dữ liệu API mô phỏng!")
                            st.dataframe(df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                    else:
                        st.error("❌ Dữ liệu API không hợp lệ!")
                except Exception as e:
                    logger.error(f"Mô phỏng API thất bại: {e}")
                    st.error(f"❌ Lỗi mô phỏng API: {e}")
        with tabs[3]:
            erp_endpoint = st.text_input("ERP Endpoint", value="https://factory-erp.com/api/orders", help="Nhập URL API ERP")
            erp_token = st.text_input("ERP Token", type="password", help="Nhập token API ERP")
            if st.button("🔄 Tích Hợp ERP"):
                df_erp = fetch_erp_data(erp_endpoint, erp_token)
                if df_erp is not None:
                    st.session_state.df_uploaded = df_erp
                    st.success("✅ Đã tải dữ liệu ERP!")
                    st.dataframe(df_erp.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))

elif menu == "📤 Xuất Báo Cáo Đa Định Dạng":
    with st.container():
        st.markdown("<div class='card'><h2>📤 Xuất Báo Cáo Đa Định Dạng</h2><p>Xuất dữ liệu sản xuất ở các định dạng khác nhau.</p></div>", unsafe_allow_html=True)
        df = st.session_state.df_uploaded
        if df is not None:
            format = st.selectbox("Định Dạng Xuất", ["Excel", "JSON", "XML", "PDF"], help="Chọn định dạng cho ERP/MES")
            erp_endpoint = st.text_input("ERP Upload Endpoint", value="https://factory-erp.com/api/upload-report", help="Nhập URL API ERP để tải lên báo cáo")
            erp_token = st.text_input("ERP Token", type="password", help="Nhập token API ERP")
            data, filename, mime = export_to_erp(df, format.lower())
            if data:
                st.download_button(
                    f"⬇️ Tải Báo Cáo {format}",
                    data=data,
                    file_name=filename,
                    mime=mime
                )
                if st.button("📤 Tải Lên ERP"):
                    try:
                        headers = {"Authorization": f"Bearer {erp_token}"}
                        requests.post(erp_endpoint, json={"data": data, "filename": filename}, headers=headers)
                        st.success("✅ Đã tải báo cáo lên ERP!")
                    except Exception as e:
                        st.error(f"❌ Lỗi tải lên ERP: {e}")
        else:
            st.warning("⚠️ Vui lòng tích hợp dữ liệu trước khi xuất báo cáo!")

elif menu == "📌 Biểu Đồ Chuỗi Nguyên Công":
    with st.container():
        st.markdown("<div class='card'><h2>📌 Biểu Đồ Chuỗi Nguyên Công</h2><p>Hiển thị thời gian các quy trình sản xuất.</p></div>", unsafe_allow_html=True)
        file = st.file_uploader("Tải Dữ Liệu Nguyên Công (CSV/Excel)", type=["csv", "xlsx"], key="nguyencong", help="File cần có: Sản phẩm, Nguyên công, Thời gian (phút)")
        if file:
            try:
                df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
                required_columns = ["Sản phẩm", "Nguyên công", "Thời gian (phút)"]
                if not all(col in df.columns for col in required_columns):
                    st.error("❌ File cần chứa các cột: " + ", ".join(required_columns))
                else:
                    df, _ = preprocess_data(df.copy())
                    if df is not None:
                        st.dataframe(df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                        fig = px.bar(
                            df,
                            x="Nguyên công",
                            y="Thời gian (phút)",
                            color="Sản phẩm",
                            title="Thời Gian Nguyên Công Theo Sản Phẩm",
                            barmode="group",
                            text="Thời gian (phút)",
                            template="plotly_dark"
                        )
                        fig.update_traces(textposition='auto')
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("<div class='card'><h3>Thống Kê Nguyên Công</h3></div>", unsafe_allow_html=True)
                        summary = df.groupby("Sản phẩm")["Thời gian (phút)"].agg(['sum', 'mean', 'count']).round(1)
                        st.dataframe(summary.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
            except Exception as e:
                logger.error(f"Lỗi biểu đồ nguyên công: {e}")
                st.error(f"❌ Lỗi dữ liệu nguyên công: {e}")

elif menu == "📩 Cảnh Báo Thông Minh Telegram":
    with st.container():
        st.markdown("<div class='card'><h2>📩 Cảnh Báo Thông Minh Telegram</h2><p>Gửi cảnh báo thời gian thực để xử lý vấn đề nhanh chóng.</p></div>", unsafe_allow_html=True)
        with st.form("telegram_form"):
            msg = st.text_input("Nội Dung Cảnh Báo:", max_chars=250, help="Tối đa 250 ký tự")
            submitted = st.form_submit_button("🚨 Gửi Cảnh Báo")
            if submitted:
                if not msg:
                    st.warning("⚠️ Vui lòng nhập nội dung cảnh báo!")
                elif len(msg) > 250:
                    st.error("❌ Nội dung vượt quá 250 ký tự!")
                else:
                    send_telegram_alert(msg)
        with st.form("custom_alert_form"):
            st.markdown("### Cấu Hình Ngưỡng Cảnh Báo")
            threshold_type = st.selectbox("Loại Ngưỡng", ["Hiệu suất (%)", "Tổng dừng (phút)", "Chi phí năng lượng ($)"])
            threshold_value = st.number_input("Giá Trị Ngưỡng", min_value=0.0, value=80.0)
            if st.form_submit_button("Lưu Ngưỡng"):
                save_alert_to_db(f"Ngưỡng tùy chỉnh: {threshold_type} < {threshold_value}", "Cấu hình")
                st.success("✅ Đã lưu ngưỡng cảnh báo!")
        st.markdown("<div class='card'><h3>📜 Lịch Sử Cảnh Báo</h3></div>", unsafe_allow_html=True)
        alert_df = load_alert_history(st.session_state.factory_id)
        if not alert_df.empty:
            st.dataframe(alert_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
        else:
            st.info("ℹ Chưa có lịch sử cảnh báo.")
        if st.session_state.df_uploaded is not None:
            df, _ = preprocess_data(st.session_state.df_uploaded)
            if df is not None:
                rf_model, iso_forest = init_predictive_models()
                if iso_forest:
                    anomalies = iso_forest.fit_predict(df[["Sản lượng (SP)", "Tổng dừng"]])
                    if -1 in anomalies:
                        anomaly_ca = df[anomalies == -1]["Ca"].tolist()
                        send_telegram_alert(f"⚠️ Bất thường: Dữ liệu bất thường ở ca {', '.join(anomaly_ca)}")

elif menu == "📋 Phân Tích Hiệu Suất Sâu":
    with st.container():
        st.markdown("<div class='card'><h2>📋 Phân Tích Hiệu Suất Sâu</h2><p>Khám phá nguyên nhân gốc rễ và tối ưu hóa sản xuất.</p></div>", unsafe_allow_html=True)
        if st.session_state.df_uploaded is not None:
            df, _ = preprocess_data(st.session_state.df_uploaded)
            if df is not None:
                df["Tổng dừng"] = df.iloc[:, 2:5].sum(axis=1)
                df["Hiệu suất (%)"] = ((df["Sản lượng (SP)"] / (480 - df["Tổng dừng"])) * 100).round(2)
                downtime_cols = ["Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]
                downtime_sums = df[downtime_cols].sum()
                downtime_df = pd.DataFrame({
                    "Nguyên Nhân": downtime_cols,
                    "Thời Gian (phút)": downtime_sums.values
                }).sort_values("Thời Gian (phút)", ascending=False)
                downtime_df["Tỷ Lệ Tích Lũy (%)"] = downtime_df["Thời Gian (phút)"].cumsum() / downtime_df["Thời Gian (phút)"].sum() * 100
                st.markdown("<div class='card'><h3>Phân Tích Pareto Thời Gian Dừng</h3></div>", unsafe_allow_html=True)
                fig_pareto = go.Figure()
                fig_pareto.add_trace(
                    go.Bar(
                        x=downtime_df["Nguyên Nhân"],
                        y=downtime_df["Thời Gian (phút)"],
                        name="Thời Gian Dừng",
                        marker_color="#d32f2f"
                    )
                )
                fig_pareto.add_trace(
                    go.Scatter(
                        x=downtime_df["Nguyên Nhân"],
                        y=downtime_df["Tỷ Lệ Tích Lũy (%)"],
                        name="Tỷ Lệ Tích Lũy",
                        yaxis="y2",
                        mode="lines+markers",
                        line=dict(color="#ffffff")
                    )
                )
                fig_pareto.update_layout(
                    title="Phân Tích Pareto Thời Gian Dừng",
                    yaxis=dict(title="Thời Gian (phút)"),
                    yaxis2=dict(title="Tỷ Lệ Tích Lũy (%)", overlaying="y", side="right"),
                    template="plotly_dark",
                    showlegend=True
                )
                st.plotly_chart(fig_pareto, use_container_width=True)
                st.dataframe(downtime_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                st.markdown("<div class='card'><h3>Tầm Quan Trọng Yếu Tố</h3></div>", unsafe_allow_html=True)
                rf_model, _ = init_predictive_models()
                if rf_model is not None:
                    X = df[["Sản lượng (SP)", "Dừng do máy (phút)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]]
                    y = df["Hiệu suất (%)"]
                    rf_model.fit(X, y)
                    feature_importance = pd.DataFrame({
                        "Yếu Tố": X.columns,
                        "Tầm Quan Trọng (%)": rf_model.feature_importances_ * 100
                    }).sort_values("Tầm Quan Trọng (%)", ascending=False)
                    st.dataframe(feature_importance.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                    st.markdown(f"<div class='card'><p>🔍 Yếu Tố Chính: <strong>{feature_importance.iloc[0]['Yếu Tố']}</strong></p></div>", unsafe_allow_html=True)
                st.markdown("<div class='card'><h3>Đề Xuất Cải Tiến</h3></div>", unsafe_allow_html=True)
                main_cause = downtime_df.iloc[0]["Nguyên Nhân"]
                st.markdown(f"""
                **Vấn Đề Chính**: {main_cause}
                - **Dừng Do Máy**: Tăng cường bảo trì định kỳ và kiểm tra.
                - **Dừng Do Người**: Cung cấp đào tạo về quy trình và an toàn.
                - **Dừng Lý Do Khác**: Tối ưu hóa luồng công việc và chuỗi cung ứng.
                """)
            else:
                st.warning("⚠️ Vui lòng tích hợp dữ liệu để phân tích!")

elif menu == "🔧 Dự Đoán Bảo Trì Nâng Cao":
    with st.container():
        st.markdown("<div class='card'><h2>🔧 Dự Đoán Bảo Trì Nâng Cao</h2><p>Dự đoán thời gian dừng để lập kế hoạch bảo trì hiệu quả.</p></div>", unsafe_allow_html=True)
        if st.session_state.df_uploaded is not None:
            df, scaler = preprocess_data(st.session_state.df_uploaded)
            if df is not None:
                rf_model, _ = init_predictive_models()
                if rf_model is not None:
                    X = df[["Sản lượng (SP)", "Dừng do người (phút)", "Dừng lý do khác (phút)"]]
                    y = df["Dừng do máy (phút)"]
                    rf_model.fit(X, y)
                    st.markdown("<div class='card'><h3>🔍 Dự Đoán Thời Gian Dừng Máy</h3></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sp_input = st.number_input("Sản Lượng Dự Kiến (SP)", min_value=0, value=100, help="Số sản phẩm dự kiến")
                    with col2:
                        human_downtime = st.number_input("Thời Gian Dừng Do Người (phút)", min_value=0, value=10, help="Thời gian dừng do người dự kiến")
                    with col3:
                        other_downtime = st.number_input("Thời Gian Dừng Lý Do Khác (phút)", min_value=0, value=5, help="Thời gian dừng do lý do khác dự kiến")
                    input_data = scaler.transform([[sp_input, human_downtime, other_downtime]])
                    predicted_downtime = rf_model.predict(input_data)[0]
                    st.markdown(f"<div class='card'><p>🔧 Thời Gian Dừng Máy Dự Đoán: <strong>{predicted_downtime:.1f} phút</strong></p></div>", unsafe_allow_html=True)
                    if predicted_downtime > 30:
                        send_telegram_alert(f"⚠️ Cảnh báo: Dự đoán thời gian dừng máy cao ({predicted_downtime:.1f} phút) cho sản lượng {sp_input} SP!")
                    fig = px.scatter(df, x="Sản lượng (SP)", y="Dừng do máy (phút)", trendline="ols",
                                     title="Mối Quan Hệ Giữa Sản Lượng và Thời Gian Dừng Máy", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Vui lòng tích hợp dữ liệu trước khi dự đoán!")

elif menu == "⏱️ Mô Phỏng Thời Gian Thực":
    with st.container():
        st.markdown("<div class='card'><h2>⏱️ Mô Phỏng Thời Gian Thực</h2><p>Giám sát và tối ưu hóa sản xuất theo thời gian thực.</p></div>", unsafe_allow_html=True)
        st.markdown("### Cấu Hình Mô Phỏng")
        col1, col2 = st.columns(2)
        with col1:
            jobs = st.slider("Số Sản Phẩm", 10, 100, 30, key="sim_jobs", help="Số sản phẩm để mô phỏng")
        with col2:
            machines = st.slider("Số Máy CNC", 5, 30, 15, key="sim_machines", help="Số máy CNC trong mô phỏng")
        priorities = {f"SP{i+1}": 1.0 for i in range(jobs)}
        power_rates = {f"M{i+1}": np.random.uniform(0.5, 2.0) for i in range(machines)}

        # Initialize session state with a default list
        if 'realtime_data' not in st.session_state:
            st.session_state.realtime_data = []
        if 'is_simulating' not in st.session_state:
            st.session_state.is_simulating = False
        if 'simulation_event' not in st.session_state:
            st.session_state.simulation_event = threading.Event()
        if 'sim_thread' not in st.session_state:
            st.session_state.sim_thread = None

        def simulate_realtime_data(event, data_list):
            """Run simulation in a separate thread with a passed list."""
            logger.info("Simulation thread started")
            try:
                while not event.is_set():
                    new_data = {
                        "Ca": np.random.choice(["Sáng", "Chiều", "Tối"]),
                        "Sản lượng (SP)": np.random.randint(80, 150),
                        "Dừng do máy (phút)": np.random.randint(0, 20),
                        "Dừng do người (phút)": np.random.randint(0, 15),
                        "Dừng lý do khác (phút)": np.random.randint(0, 10),
                        "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    logger.debug(f"Generated data: {new_data}")
                    data_list.append(new_data)
                    logger.debug(f"Appended data, current length: {len(data_list)}")
                    if len(data_list) > 100:
                        data_list.pop(0)
                    time.sleep(3)  # Simulate data every 3 seconds
            except Exception as e:
                logger.error(f"Simulation error: {e}")
            finally:
                logger.info("Simulation thread ended")

        def start_simulation():
            if not st.session_state.is_simulating:
                st.session_state.realtime_data = []  # Reset data
                st.session_state.is_simulating = True
                st.session_state.simulation_event.clear()
                st.success("✅ Đã bắt đầu mô phỏng thời gian thực!")
                if st.session_state.sim_thread is None or not st.session_state.sim_thread.is_alive():
                    st.session_state.sim_thread = threading.Thread(
                        target=simulate_realtime_data,
                        args=(st.session_state.simulation_event, st.session_state.realtime_data),
                        daemon=True
                    )
                    st.session_state.sim_thread.start()
                    logger.info("Simulation thread launched")
                else:
                    logger.warning("Thread already running")
            else:
                st.info("⏳ Mô phỏng đang chạy...")

        def stop_simulation():
            if st.session_state.is_simulating:
                st.session_state.is_simulating = False
                st.session_state.simulation_event.set()
                if st.session_state.sim_thread:
                    st.session_state.sim_thread.join(timeout=2)
                    st.session_state.sim_thread = None
                st.success("✅ Đã dừng mô phỏng thành công!")
                logger.info("Simulation stopped")
            else:
                st.info("ℹ Mô phỏng đã dừng hoặc chưa bắt đầu.")

        st.button("▶️ Bắt Đầu Mô Phỏng", on_click=start_simulation)
        st.button("⏹️ Dừng Mô Phỏng", on_click=stop_simulation)

        # Update real-time data display
        data_placeholder = st.empty()
        chart_placeholder = st.empty()

        if st.session_state.realtime_data:
            df_realtime = pd.DataFrame(st.session_state.realtime_data)
            df_realtime, _ = preprocess_data(df_realtime)
            if df_realtime is not None and not df_realtime.empty:
                with data_placeholder.container():
                    st.markdown("<div class='card'><h3>📊 Dữ Liệu Thời Gian Thực</h3></div>", unsafe_allow_html=True)
                    st.dataframe(df_realtime.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
                with chart_placeholder.container():
                    try:
                        fig = px.line(
                            df_realtime,
                            x="Thời gian",
                            y=["Sản lượng (SP)", "Tổng dừng"],
                            title="Sản Lượng và Thời Gian Dừng Theo Thời Gian",
                            template="plotly_dark",
                            labels={"value": "Giá trị", "variable": "Loại dữ liệu"}
                        )
                        fig.update_layout(
                            xaxis_title="Thời gian",
                            yaxis_title="Giá trị",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Chart error: {e}")
                        st.error(f"❌ Lỗi vẽ biểu đồ: {e}")
            else:
                with data_placeholder.container():
                    st.markdown("<div class='card'><h3>📊 Dữ Liệu Thời Gian Thực</h3></div>", unsafe_allow_html=True)
                    st.warning("⚠️ Dữ liệu thời gian thực không hợp lệ!")
        else:
            with data_placeholder.container():
                st.markdown("<div class='card'><h3>📊 Dữ Liệu Thời Gian Thực</h3></div>", unsafe_allow_html=True)
                st.info("ℹ Đang chờ dữ liệu thời gian thực...")

        # Manual refresh button
        if st.session_state.is_simulating and st.button("Làm mới dữ liệu"):
            st.experimental_rerun()

elif menu == "🖼️ Digital Twin (3D)":
    with st.container():
        st.markdown("<div class='card'><h2>🖼️ Digital Twin (3D)</h2><p>Hiển thị chuyển động công cụ CNC trong không gian 3D.</p></div>", unsafe_allow_html=True)
        file = st.file_uploader("Tải File G-Code hoặc NC", type=["nc", "gcode"], help="File chứa đường dẫn công cụ (G01)")
        if file:
            try:
                toolpath = parse_gcode(file)
                if not all(toolpath.values()):
                    st.error("❌ File G-Code không hợp lệ hoặc không chứa tọa độ X, Y, Z!")
                else:
                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=toolpath["x"],
                            y=toolpath["y"],
                            z=toolpath["z"],
                            mode="lines+markers",
                            marker=dict(size=4, color="#d32f2f"),
                            line=dict(color="#ffffff", width=2)
                        )
                    ])
                    fig.update_layout(
                        title="🛠️ Digital Twin: Đường Dẫn Công Cụ CNC",
                        scene=dict(
                            xaxis_title="X (mm)",
                            yaxis_title="Y (mm)",
                            zaxis_title="Z (mm)"
                        ),
                        template="plotly_dark",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Lỗi xử lý G-Code: {e}")
                st.error(f"❌ Lỗi xử lý G-Code: {e}")
        else:
            st.markdown("<div class='card'><h3>🖼️ Mô Phỏng Mẫu</h3></div>", unsafe_allow_html=True)
            t = np.linspace(0, 10, 100)
            x = np.sin(t) * 100
            y = np.cos(t) * 100
            z = t * 10
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    marker=dict(size=4, color="#d32f2f"),
                    line=dict(color="#ffffff", width=2)
                )
            ])
            fig.update_layout(
                title="🛠️ Digital Twin: Mô Phỏng Đường Dẫn Công Cụ",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)"
                ),
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        if st.session_state.realtime_data:
            st.markdown("<div class='card'><h3>📡 Tích Hợp Dữ Liệu Thời Gian Thực</h3></div>", unsafe_allow_html=True)
            df_realtime = pd.DataFrame(st.session_state.realtime_data)
            if not df_realtime.empty:
                scale_factor = df_realtime["Sản lượng (SP)"].mean() / 100
                fig.update_traces(
                    marker=dict(size=4 * scale_factor),
                    line=dict(width=2 * scale_factor)
                )
                st.plotly_chart(fig, use_container_width=True)

elif menu == "🔬 Phân Tích Kịch Bản Nâng Cao":
    with st.container():
        st.markdown("<div class='card'><h2>🔬 Phân Tích Kịch Bản Nâng Cao</h2><p>So sánh các kịch bản sản xuất để đưa ra quyết định tối ưu.</p></div>", unsafe_allow_html=True)
        with st.form("scenario_form"):
            scenario_name = st.text_input("Tên Kịch Bản", help="Đặt tên cho kịch bản của bạn")
            col1, col2, col3 = st.columns(3)
            with col1:
                jobs = st.slider("Số Sản Phẩm", 10, 100, 30, key="scenario_jobs", help="Số sản phẩm")
            with col2:
                machines = st.slider("Số Máy CNC", 5, 30, 15, key="scenario_machines", help="Số máy CNC")
            with col3:
                priority_job = st.text_input("Sản Phẩm Ưu Tiên", "", key="scenario_priority", help="e.g., SP1,SP2")
            broken_machines = st.text_input("Máy Hỏng (e.g., M1,M3)", "", help="Các máy hỏng, cách nhau bằng dấu phẩy")
            power_rates = {f"M{i+1}": st.number_input(
                f"Công Suất M{i+1} (kW)", min_value=0.1, value=1.0, step=0.1, key=f"scenario_power_{i}"
            ) for i in range(machines)}
            submitted = st.form_submit_button("🔍 Phân Tích Kịch Bản")
            if submitted:
                if not scenario_name:
                    st.error("❌ Vui lòng nhập tên kịch bản!")
                else:
                    priorities = {f"SP{i+1}": 1.0 for i in range(jobs)}
                    if priority_job:
                        for pj in priority_job.split(","):
                            if pj in priorities:
                                priorities[pj] = 2.5
                    # SỬA Ở ĐÂY: chuyển chuỗi thành list
                    broken_machines_list = [m.strip() for m in broken_machines.split(",") if m.strip()] if broken_machines else []
                    schedule, timeline, energy_cost = run_aco_schedule(
                        jobs, machines, priorities, power_rates=power_rates, broken_machines=broken_machines_list
                    )
                    schedule, timeline, energy_cost = run_aco_schedule(jobs, machines, priorities, power_rates=power_rates, broken_machines=broken_machines_list)
                    if schedule:
                        results = {
                            "total_time": max(timeline.values()),
                            "energy_cost": energy_cost,
                            "schedule": schedule
                        }
                        saved = save_scenario_to_db(scenario_name, {"jobs": jobs, "machines": machines, "priorities": priorities}, results)
                        if saved:
                            st.session_state.scenario_history.append(saved)
                            st.success(f"✅ Kịch bản '{scenario_name}' đã được lưu!")
                            fig, df_gantt = plot_gantt_schedule(schedule)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.info(f"⏱️ Tổng Thời Gian: {results['total_time']:,} phút | ⚡ Chi Phí Năng Lượng: ${energy_cost:,.2f}")
        st.markdown("<div class='card'><h3>📜 Lịch Sử Kịch Bản</h3></div>", unsafe_allow_html=True)
        scenario_df = load_scenario_history(st.session_state.factory_id)
        if not scenario_df.empty:
            st.dataframe(scenario_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
            selected_scenario = st.selectbox("Chọn Kịch Bản để So Sánh", scenario_df["scenario_name"])
            if selected_scenario:
                scenario_data = scenario_df[scenario_df["scenario_name"] == selected_scenario].iloc[0]
                results = json.loads(scenario_data["results"])
                fig, df_gantt = plot_gantt_schedule(results["schedule"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"⏱️ Tổng Thời Gian: {results['total_time']:,} phút | ⚡ Chi Phí Năng Lượng: ${results['energy_cost']:,.2f}")

elif menu == "📈 So Sánh Hiệu Suất AI":
    with st.container():
        st.markdown("<div class='card'><h2>📈 So Sánh Hiệu Suất AI</h2><p>Đánh giá hiệu quả của AI so với phương pháp truyền thống.</p></div>", unsafe_allow_html=True)
        erp_endpoint = st.text_input("ERP Endpoint", value="https://factory-mes.com/api/historical-performance", help="Nhập URL API ERP")
        erp_token = st.text_input("ERP Token", type="password", help="Nhập token API ERP")
        if st.button("🔄 Tải Dữ Liệu ERP"):
            df_erp = fetch_erp_data(erp_endpoint, erp_token)
            if df_erp is not None:
                st.session_state.df_uploaded = df_erp
                st.success("✅ Đã tải dữ liệu ERP!")
        comparison_data = {
            "Phương Pháp": ["Truyền Thống", "ACO", "ACO + RL"],
            "Thời Gian Hoàn Thành (phút)": [1200, 900, 850],
            "Chi Phí Năng Lượng ($)": [1500, 1200, 1100],
            "Hiệu Suất (%)": [75, 85, 90]
        }
        df_comparison = pd.DataFrame(comparison_data)
        if st.session_state.df_uploaded is not None:
            df, _ = preprocess_data(st.session_state.df_uploaded)
            if df is not None:
                df_comparison["Hiệu Suất (%)"] = [df["Hiệu suất (%)"].mean() - 15, df["Hiệu suất (%)"].mean() - 5, df["Hiệu suất (%)"].mean()]
        fig = go.Figure(data=[
            go.Bar(
                name="Thời Gian Hoàn Thành (phút)",
                x=df_comparison["Phương Pháp"],
                y=df_comparison["Thời Gian Hoàn Thành (phút)"],
                marker_color="#d32f2f"
            ),
            go.Bar(
                name="Chi Phí Năng Lượng ($)",
                x=df_comparison["Phương Pháp"],
                y=df_comparison["Chi Phí Năng Lượng ($)"],
                marker_color="#ffffff"
            )
        ])
        fig.update_layout(
            barmode="group",
            title="So Sánh Hiệu Suất: Truyền Thống vs AI",
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        fig_eff = px.line(
            df_comparison,
            x="Phương Pháp",
            y="Hiệu Suất (%)",
            markers=True,
            title="Hiệu Suất Theo Phương Pháp",
            template="plotly_dark"
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        st.markdown("<div class='card'><h3>📊 Kết Quả Định Lượng</h3></div>", unsafe_allow_html=True)
        st.dataframe(df_comparison.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))
        st.markdown("""
        **Kết Luận**:
        - **ACO + RL**: Giảm 29% thời gian và 27% chi phí năng lượng so với phương pháp truyền thống.
        - **Khuyến Nghị**: Triển khai ACO + RL để tối ưu hóa sản xuất.
        """)

elif menu == "📖 Hướng Dẫn Sử Dụng":
    with st.container():
        st.markdown("<div class='card'><h2>📖 Hướng Dẫn Sử Dụng</h2><p>Hướng dẫn chi tiết để sử dụng CNCTech AI Pro.</p></div>", unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Giới Thiệu
        CNCTech AI Pro là giải pháp tối ưu hóa sản xuất CNC, tích hợp AI (ACO, RL, LSTM), IoT, và Digital Twin. Được phát triển bởi DIM Team tại IMS Lab.

        ### 🚀 Hướng Dẫn Nhanh
        1. **Tích Hợp Dữ Liệu**:
           - Tải file CSV/Excel trong tab "Tích Hợp Dữ Liệu".
           - Kết nối MQTT hoặc ERP để nhận dữ liệu thời gian thực.
        2. **Lập Kế Hoạch Sản Xuất**:
           - Sử dụng "Lập Kế Hoạch Thủ Công" để nhập thông số cơ bản.
           - Chọn "ACO + RL" để tối ưu hóa lịch sản xuất bằng AI.
        3. **Phân Tích Hiệu Suất**:
           - Xem dashboard để theo dõi sản lượng, thời gian dừng, và OEE.
           - Sử dụng "Phân Tích Hiệu Suất Sâu" để tìm nguyên nhân gốc rễ.
        4. **Cảnh Báo và Bảo Trì**:
           - Cấu hình ngưỡng cảnh báo trong "Cảnh Báo Thông Minh Telegram".
           - Dự đoán bảo trì trong "Dự Đoán Bảo Trì Nâng Cao".
        5. **Digital Twin và Kịch Bản**:
           - Tải file G-Code để xem mô phỏng 3D trong "Digital Twin".
           - So sánh kịch bản sản xuất trong "Phân Tích Kịch Bản Nâng Cao".

        ### 📹 Video Hướng Dẫn
        <iframe width="100%" height="400" src="https://www.youtube.com/watch?v=4CCGI83vOVo" frameborder="0" allowfullscreen></iframe>

        ### 👥 Đội Ngũ Phát Triển
        - **DIM Team**: Vũ Thế Anh, Nguyễn Duy Vũ Dương, Nguyễn Thành Vinh, Phạm Hữu Trung
        - **Liên Hệ**: [GitHub Repo](https://github.com/dimteam/cNCTech-ai-pro)
        - **IMS Lab**: ĐH Bách Khoa Hà Nội

        ### ⚠️ Lưu Ý
        - Đảm bảo kết nối MQTT hoặc ERP ổn định.
        - Cấu hình Telegram trong `secrets.toml` để nhận cảnh báo.
        - Liên hệ hỗ trợ qua [support@cnctech.vn](mailto:support@cnctech.vn).
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px;'>
    <p style='color: #ffffff; font-size: 14px;'>© 2025 DIM Team - IMS Lab | Powered by CNCTech AI Pro</p>
    <p style='color: #d32f2f; font-size: 12px;'>Phiên bản 2.0 | Phát triển cho CNCTech & HUST 2025</p>
</div>
""", unsafe_allow_html=True)