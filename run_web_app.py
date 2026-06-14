import os
import sys
import sqlite3

# Chuyển thư mục làm việc về thư mục chứa script để tránh lỗi đường dẫn tương đối
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Đảm bảo stdout/stderr sử dụng UTF-8 để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import json
import urllib.request
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import ml_pipeline

def load_env_values():
    """Tự động nạp API Key từ file .env cục bộ và xóa các biến cũ để tránh cache."""
    # Luôn xóa các biến môi trường cũ trước để tránh rác từ môi trường cha
    for key in list(os.environ.keys()):
        if key in ['GEMINI_API_KEY', 'GEMINI_MODEL', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY'] or key.startswith('GEMINI_API_KEY_BACKUP'):
            del os.environ[key]
            
    # Định vị đường dẫn tuyệt đối tới file .env dựa trên thư mục của script này
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
        except Exception as e:
            print(f"Warning: Lỗi đọc file .env tại {env_path}: {e}")

# Nạp môi trường lần đầu khi start server
load_env_values()

def get_gemini_api_keys():
    """Lấy danh sách các API Key khả dụng từ môi trường (bao gồm cả backup)."""
    load_env_values()
    keys = []
    
    # 1. Lấy key chính
    main_key = os.environ.get('GEMINI_API_KEY')
    if main_key and main_key.strip() and not main_key.startswith("YOUR_"):
        keys.append(main_key.strip())
        
    # 2. Lấy các key backup (GEMINI_API_KEY_BACKUP, GEMINI_API_KEY_BACKUP_2, ...)
    backup_keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY_BACKUP") and v.strip() and not v.startswith("YOUR_"):
            backup_keys.append((k, v.strip()))
            
    # Sắp xếp theo tên biến để đúng thứ tự ưu tiên
    backup_keys.sort(key=lambda x: x[0])
    for k, val in backup_keys:
        keys.append(val)
        
    return keys

# --- CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE (LƯU LỊCH SỬ BỆNH NHÂN) ---
DATABASE_PATH = 'models/amr_history.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Xóa bảng cũ để cập nhật cấu trúc schema mới (strain_id thay vì patient_id)
        cursor.execute('DROP TABLE IF EXISTS prediction_history')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strain_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                detected_genes TEXT,
                features_json TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("SQLite Database initialized successfully with strain_id schema.")
        
        # Gọi seed_mock_data để khởi tạo dữ liệu giả nếu database trống
        seed_mock_data()
    except Exception as e:
        print(f"Error initializing SQLite Database: {e}")

def seed_mock_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Kiểm tra xem bảng đã có dữ liệu chưa
        cursor.execute('SELECT COUNT(*) FROM prediction_history')
        count = cursor.fetchone()[0]
        if count >= 5:
            conn.close()
            return
        
        print("Seeding mock data for epidemiology dashboard...")
        import datetime
        import random
        import json
        
        # Danh sách gen mẫu từ GENE_DB
        genes_pool = ["blaCTX-M-15", "gyrA_S83L", "floR", "tet(A)", "sul1", "parC_S80I", "dfrA17", "blaTEM-1"]
        
        # Tạo 15 bản ghi lịch sử trong 15 ngày qua
        now = datetime.datetime.now()
        for i in range(15, 0, -1):
            # Ngày cách hiện tại i ngày
            date_time = now - datetime.timedelta(days=i, hours=random.randint(0, 12), minutes=random.randint(0, 59))
            date_str = date_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Auto strain ID (EC đại diện cho Escherichia coli)
            strain_id = f"EC-202606-{100 + i}"
            
            # Chọn ngẫu nhiên vài gen kháng
            num_genes = random.randint(0, 4)
            chosen_genes = random.sample(genes_pool, num_genes) if num_genes > 0 else []
            detected_genes_str = ", ".join(chosen_genes) if chosen_genes else "Không phát hiện"
            
            # Quy định xác suất kháng và kết luận tương quan (Đột biến gyrA_S83L gây kháng mạnh Ciprofloxacin)
            if num_genes >= 2 or ("blaCTX-M-15" in chosen_genes) or ("gyrA_S83L" in chosen_genes):
                prediction = "Resistant"
                probability = round(random.uniform(0.53, 0.98), 4)
            else:
                prediction = "Susceptible"
                probability = round(random.uniform(0.05, 0.51), 4)
                
            # Tạo mock features_json
            mock_features = {}
            for g in genes_pool:
                mock_features[g] = 1.0 if g in chosen_genes else 0.0
            # Thêm k-mer nền giả lập
            for k in range(10):
                mock_features[f"kmer_{k}"] = round(random.uniform(0.0, 5.0), 3)
                
            cursor.execute('''
                INSERT INTO prediction_history (timestamp, strain_id, prediction, probability, detected_genes, features_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                strain_id,
                prediction,
                probability,
                detected_genes_str,
                json.dumps(mock_features)
            ))
        
        conn.commit()
        conn.close()
        print("Mock data seeded successfully for epidemiology dashboard.")
    except Exception as e:
        print(f"Error seeding mock data: {e}")

# --- CƠ SỞ DỮ LIỆU LUẬT KHÁNG THUỐC CỦA GEN (GENE KNOWLEDGE BASE) ---
GENE_DB = {
    # --- CÁC ĐỘT BIẾN GEN & KHÁNG THUỐC ---
    "gyrA_D87N": "Đột biến điểm trong gyrA làm thay đổi cấu trúc enzyme DNA gyrase, trực tiếp gây đề kháng kháng sinh nhóm Fluoroquinolone.",
    "gyrA_S83L": "Đột biến điểm trong gyrA, là nguyên nhân phổ biến nhất gây đề kháng mạnh kháng sinh nhóm Fluoroquinolone (Ciprofloxacin, Levofloxacin).",
    "floR": "Gen mã hóa protein bơm đẩy (efflux pump), gây đề kháng thuốc Florfenicol và Chloramphenicol.",
    "tet(A)": "Gen kháng Tetracycline hoạt động theo cơ chế bơm đẩy chủ động loại A.",
    "tet(B)": "Gen kháng Tetracycline hoạt động theo cơ chế bơm đẩy chủ động loại B.",
    "aph(6)-Id": "Enzyme aminoglycoside phosphotransferase kháng kháng sinh nhóm Aminoglycoside (Streptomycin).",
    "aph(3'')-Ib": "Enzyme aminoglycoside phosphotransferase kháng kháng sinh nhóm Aminoglycoside.",
    "sul2": "Gen kháng thuốc diệt khuẩn nhóm Sulfonamide theo cơ chế thay thế mục tiêu enzyme DHPS.",
    "sul1": "Gen kháng Sulfonamide thường đi kèm integron lớp 1.",
    "parC_S80I": "Đột biến trong parC (topoisomerase IV) kết hợp đột biến gyrA làm tăng vọt mức đề kháng kháng sinh nhóm Fluoroquinolone.",
    "dfrA17": "Gen kháng Trimethoprim theo cơ chế thay thế mục tiêu enzyme dihydrofolate reductase (DHFR).",
    "aadA5": "Gen kháng kháng sinh Streptomycin và Spectinomycin.",
    "blaCTX-M-15": "Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) nhóm CTX-M, gây đề kháng mạnh toàn bộ các kháng sinh nhóm Cephalosporin thế hệ 3, thế hệ 4 (như Ceftriaxone, Cefotaxime, Cefepime) và Monobactam.",
    "blaCTX-M-14": "Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) kháng kháng sinh Cephalosporin.",
    "blaTEM-1": "Beta-lactamase phổ hẹp, kháng các kháng sinh nhóm Penicillin và Cephalosporin thế hệ 1.",
    "qacEdelta1": "Gen kháng các chất khử trùng bậc bốn (quaternary ammonium compounds) dùng trong môi trường y tế.",

    # --- THUẬT NGỮ LÂM SÀNG & KHÁNG SINH ĐỒ ---
    "antibiogram": "Kháng sinh đồ (Antibiogram): Phương pháp thử nghiệm trong phòng thí nghiệm để đo lường mức độ nhạy cảm của vi khuẩn đối với các loại kháng sinh khác nhau, từ đó giúp bác sĩ lựa chọn phác đồ điều trị tối ưu.",
    "amr": "Antimicrobial Resistance (Kháng kháng sinh): Hiện tượng vi sinh vật (như vi khuẩn, virus, nấm) biến đổi để chống lại tác dụng của thuốc điều trị, làm các phương pháp điều trị thông thường mất tác dụng.",
    "mic": "Minimum Inhibitory Concentration (Nồng độ ức chế tối thiểu - MIC): Nồng độ kháng sinh thấp nhất có khả năng ức chế sự phát triển rõ rệt của vi khuẩn sau một thời gian nuôi cấy.",
    "esbl": "Extended-Spectrum Beta-Lactamase (Beta-lactamase phổ rộng): Enzyme do vi khuẩn sinh ra làm bất hoạt hầu hết kháng sinh nhóm beta-lactam phổ rộng như Cephalosporin thế hệ 3, 4.",
    "efflux pump": "Bơm đẩy chủ động (Efflux Pump): Cơ chế đề kháng của vi khuẩn bằng cách chủ động bơm kháng sinh ra khỏi tế bào vi sinh vật, làm giảm nồng độ thuốc bên trong vi khuẩn.",
    "beta-lactamase": "Beta-lactamase: Nhóm enzyme do vi khuẩn sinh ra để phá hủy cấu trúc vòng beta-lactam của kháng sinh (như Penicillin, Cephalosporin), làm vô hiệu hóa hoạt tính của thuốc.",
    
    # --- CÁC NHÓM KHÁNG SINH CHÍNH ---
    "fluoroquinolone": "Fluoroquinolone: Nhóm kháng sinh diệt khuẩn phổ rộng (như Ciprofloxacin, Levofloxacin), hoạt động bằng cách ức chế quá trình tổng hợp DNA của vi khuẩn.",
    "cephalosporin": "Cephalosporin: Nhóm kháng sinh beta-lactam diệt khuẩn phổ rộng, gồm nhiều thế hệ (như Ceftriaxone thế hệ 3, Cefepime thế hệ 4) chuyên trị các bệnh nhiễm trùng nặng.",
    "tetracycline": "Tetracycline: Nhóm kháng sinh kìm khuẩn phổ rộng (như Tetracycline, Doxycycline), hoạt động bằng cách ức chế quá trình tổng hợp protein tại ribosome 30S.",
    "carbapenem": "Carbapenem: Nhóm kháng sinh beta-lactam phổ cực rộng và mạnh (như Imipenem, Meropenem), thường được xem là lựa chọn cuối cùng để điều trị vi khuẩn đa kháng thuốc.",
    "aminoglycoside": "Aminoglycoside: Nhóm kháng sinh diệt khuẩn mạnh (như Streptomycin, Gentamicin, Amikasin), hoạt động bằng cách gắn vào ribosome 30S để ức chế dịch mã protein.",
    "penicillin": "Penicillin: Nhóm kháng sinh beta-lactam đầu tiên của y học (như Ampicillin, Amoxicillin), hoạt động bằng cách ức chế tổng hợp vách tế bào vi khuẩn.",

    # --- THUẬT NGỮ HỌC MÁY (MACHINE LEARNING) & ĐỒ ÁN ---
    "shap": "SHAP (Shapley Additive exPlanations): Phương pháp định lượng mức độ đóng góp (tích cực hay tiêu cực) của từng đặc trưng gen/k-mer vào quyết định dự đoán kháng thuốc của mô hình.",
    "stacking": "Stacking Ensemble: Kỹ thuật học máy kết hợp nhiều mô hình nền tảng khác nhau (Random Forest, XGBoost, SVM...) để tối ưu hóa độ chính xác và độ tin cậy của chẩn đoán.",
    "k-mer": "k-mer: Đoạn con nucleotide độ dài k cố định được trích xuất từ chuỗi gen vi khuẩn, đóng vai trò làm đặc trưng đầu vào cho các thuật toán học máy dự đoán AMR.",
    "random forest": "Random Forest (Rừng ngẫu nhiên): Thuật toán học máy dựa trên tập hợp nhiều cây quyết định hoạt động độc lập, dự đoán bằng cách bỏ phiếu số đông.",
    "xgboost": "XGBoost (Extreme Gradient Boosting): Thuật toán học máy dựa trên Gradient Boosting hiệu năng cao, thường đứng đầu về tốc độ và độ chính xác trên tập dữ liệu bảng sinh học.",
    "svm": "SVM (Support Vector Machine): Thuật toán phân loại học máy hoạt động bằng cách tìm siêu phẳng tối ưu để phân tách các mẫu kháng thuốc và nhạy cảm trong không gian đa chiều.",
    "resistant": "Resistant (Kháng thuốc): Trạng thái vi khuẩn kháng lại thuốc thử nghiệm, kháng sinh này không còn hiệu quả lâm sàng cho điều trị.",
    "susceptible": "Susceptible (Nhạy cảm): Trạng thái vi khuẩn bị tiêu diệt hoặc ức chế bởi nồng độ kháng sinh thông thường, có thể điều trị thành công bằng thuốc này."
}


def generate_local_report(outcome, probability, top_features, threshold):
    """Tự động tạo báo cáo phân tích đề kháng dựa trên cơ sở dữ liệu luật gen kháng thuốc cục bộ."""
    outcome_vietnamese = "KHÁNG CIPROFLOXACIN (Resistant)" if outcome == "Resistant" else "NHẠY CẢM CIPROFLOXACIN (Susceptible)"
    
    report = "### 🧬 Báo cáo Phân tích Đề kháng Ciprofloxacin ở E. coli (AI Local Expert System)\n\n"
    report += f"- **Kết quả phân loại:** {outcome_vietnamese}\n"
    report += f"- **Xác suất Đề kháng Ciprofloxacin:** **{probability * 100:.2f}%**\n"
    report += f"- **Ngưỡng mô hình quyết định:** {threshold:.3f}\n\n"
    
    # Lọc ra các gen kháng thuốc xuất hiện trong mẫu có giá trị dương
    detected_genes = []
    for f in top_features:
        name = f['feature']
        val = f['feature_value']
        shap_val = f['shap_value']
        if name in GENE_DB and val > 0:
            detected_genes.append((name, GENE_DB[name], shap_val))
            
    if detected_genes:
        report += "#### 🧬 Phát hiện các đặc trưng kháng thuốc chủ đạo:\n"
        for name, desc, shap_val in detected_genes:
            report += f"- **{name}** (Tác động SHAP: `+{shap_val:.4f}`): {desc}\n"
        report += "\n"
    else:
        report += "#### 🧬 Phân tích cấu trúc k-mer & Hệ gen vi khuẩn:\n"
        report += "Không phát hiện thấy gen kháng thuốc AMR điển hình hoạt động ở mức biểu hiện dương tính cao. Kết quả phân loại được thúc đẩy bởi sự thay đổi mật độ các k-mer nền trong hệ gen vi khuẩn.\n\n"
        
    # Khuyến nghị y khoa
    report += "#### 💊 Hướng dẫn và Đề xuất điều trị nhiễm trùng do chủng này:\n"
    if outcome == "Resistant":
        report += "1. ❌ **Hạn chế sử dụng:** Tránh sử dụng Ciprofloxacin và các kháng sinh nhóm Fluoroquinolone khác do chủng vi khuẩn này đã đề kháng cao (thúc đẩy bởi các đột biến đích như `gyrA_S83L` hoặc các gen kháng liên quan).\n"
        report += "2. 🔄 **Kháng sinh thay thế cân nhắc:** Cân nhắc sử dụng nhóm kháng sinh **Carbapenem** (Imipenem, Meropenem) hoặc phối hợp thuốc có hiệu quả vi sinh dựa trên kháng sinh đồ thực tế.\n"
        report += "3. 🔬 **Cận lâm sàng:** Tiến hành thử nghiệm kháng sinh đồ đĩa giấy khuếch tán để xác định giá trị MIC thực tế trước khi dùng phác đồ bậc cao.\n"
    else:
        report += "1. ✅ **Hướng dẫn sử dụng:** Chủng vi khuẩn E. coli nhạy cảm với Ciprofloxacin. Bác sĩ có thể tiếp tục sử dụng phác đồ Ciprofloxacin tiêu chuẩn nếu phù hợp lâm sàng để tránh lạm dụng kháng sinh thế hệ mới.\n"
        report += "2. 📈 **Theo dõi lâm sàng:** Giám sát phản ứng sốt và các chỉ số nhiễm trùng (CRP, PCT) của bệnh nhân trong 48 giờ đầu tiên để đánh giá hiệu quả đáp ứng thuốc thực tế.\n"
        
    # Khuyến nghị theo nhóm tuổi & thai kỳ
    report += "\n#### ⚠️ Lưu ý chống chỉ định lâm sàng của các kháng sinh đối với bệnh nhân nhiễm khuẩn:\n"
    report += "- **Fluoroquinolones (Ciprofloxacin, Levofloxacin):** Chống chỉ định cho **trẻ em dưới 18 tuổi** (nguy cơ tổn thương sụn khớp) và phụ nữ mang thai / cho con bú.\n"
    report += "- **Tetracyclines (Tetracycline, Doxycycline):** Chống chỉ định cho **trẻ em dưới 8 tuổi** (nguy cơ gây biến màu răng vĩnh viễn và chậm phát triển xương) và phụ nữ mang thai.\n"
    report += "- **Aminoglycosides (Streptomycin, Gentamicin):** Thận trọng đặc biệt và cần chỉnh liều ở **người cao tuổi (> 65 tuổi)** và trẻ sơ sinh (do độc tính tích lũy gây suy thận và điếc không hồi phục).\n"
    report += "- **Cephalosporins & Penicillins:** Tương đối an toàn cho trẻ em và phụ nữ mang thai, tuy nhiên cần kiểm tra kỹ tiền sử dị ứng penicillin.\n"

    return report

def generate_local_chat_reply(message, outcome, probability, top_features):
    """Phản hồi cục bộ của Trợ lý AI khi không có API Key (Local Expert Mode)."""
    msg_lower = message.lower()
    
    # 1. Trả lời câu chào hỏi
    import re
    words_set = set(re.findall(r'[a-zA-Z0-9\-_]+', msg_lower))
    # Dùng set để so khớp từ nguyên bản (tránh "hi" khớp nhầm các từ tiếng Việt như "nguy hiểm", "chỉ", "khi"...)
    if any(k in words_set for k in ["chào", "hello", "hi", "xin-chào", "xin_chào"]) or "xin chào" in msg_lower:
        return "Xin chào! Tôi là **Trợ lý Nghiên cứu AMR (Local Expert Mode)**. Tôi sẵn sàng hỗ trợ bạn giải thích các đặc trưng kiểu gen kháng thuốc Ciprofloxacin của chủng vi khuẩn E. coli. Bạn cần tôi giải thích điều gì?"

    # 2. Giải thích về các gen cụ thể
    found_genes = []
    import re
    # Trích xuất các cụm từ (bao gồm cả ký tự đặc biệt như - hoặc _) từ tin nhắn để khớp từ khóa
    msg_words = re.findall(r'[a-zA-Z0-9\-_]+', msg_lower)
    
    for gene, desc in GENE_DB.items():
        gene_lower = gene.lower()
        clean_gene = gene.replace("(", "").replace(")", "").replace("'", "").lower()
        
        # 2a. Khớp trực tiếp (Ví dụ: "blaCTX-M-15 là gì")
        if gene_lower in msg_lower or clean_gene in msg_lower:
            found_genes.append(f"- **{gene}**: {desc}")
            continue
            
        # 2b. Khớp một phần từ khóa dài hơn 3 ký tự (Ví dụ: "blaCTX-M là gì", "gyrA là gì")
        for word in msg_words:
            if len(word) >= 3 and (word in gene_lower or word in clean_gene):
                found_genes.append(f"- **{gene}**: {desc}")
                break
            
    if found_genes:
        reply = "### 🧬 Giải thích về đột biến / gen kháng thuốc được nhắc đến:\n\n"
        reply += "\n".join(found_genes)
        reply += "\n\n*Thông tin được trích xuất từ Cơ sở dữ liệu Luật kháng thuốc Ciprofloxacin của hệ thống.*"
        return reply

    # 3. Câu hỏi về phác đồ điều trị / Kháng sinh thay thế
    if any(k in msg_lower for k in ["kháng sinh", "thuốc", "phác đồ", "điều trị", "thay thế", "kê đơn", "prescribe", "treatment"]):
        if outcome == "Resistant":
            return f"### 💊 Đề xuất lâm sàng hướng điều trị (Local Expert Mode)\n\n" \
                   f"Do chủng vi khuẩn E. coli này có xác suất đề kháng Ciprofloxacin cao (**{probability * 100:.1f}%**):\n" \
                   f"1. ❌ **Hạn chế:** Không sử dụng Ciprofloxacin hoặc các kháng sinh nhóm Fluoroquinolone khác để điều trị nhiễm trùng do chủng này.\n" \
                   f"2. 🔄 **Thay thế:** Cân nhắc điều trị bằng nhóm kháng sinh thay thế như **Carbapenem** (Imipenem, Meropenem) hoặc phối hợp thuốc có độ nhạy phù hợp trên kháng sinh đồ thực tế.\n"
        else:
            return "### 💊 Đề xuất lâm sàng hướng điều trị (Local Expert Mode)\n\n" \
                   "Do chủng vi khuẩn E. coli nhạy cảm với Ciprofloxacin (Susceptible):\n" \
                   "1. ✅ **Hướng dẫn:** Có thể tiếp tục sử dụng phác đồ điều trị Ciprofloxacin tiêu chuẩn nếu phù hợp lâm sàng.\n" \
                   "2. 🔍 **Theo dõi:** Giám sát đáp ứng lâm sàng để đảm bảo bệnh nhân giảm sốt và các chỉ số nhiễm trùng đi xuống."

    # 3b. Câu hỏi về độ tuổi / thai kỳ / chống chỉ định
    if any(k in msg_lower for k in ["tuổi", "trẻ em", "phụ nữ", "mang thai", "thai kỳ", "bà bầu", "người già", "cao tuổi", "chống chỉ định"]):
        return "### ⚠️ Hướng dẫn Chống chỉ định Lâm sàng theo đối tượng (Offline Mode)\n\n" \
               "- 👶 **Trẻ em:**\n" \
               "  * **Dưới 18 tuổi:** Tránh dùng nhóm **Fluoroquinolones** (như Ciprofloxacin, Levofloxacin) do nguy cơ gây tổn thương sụn khớp ở các khớp chịu lực.\n" \
               "  * **Dưới 8 tuổi:** Tránh dùng nhóm **Tetracyclines** (như Tetracycline, Doxycycline) do nguy cơ gây biến màu răng vĩnh viễn và chậm phát triển xương.\n" \
               "- 🤰 **Phụ nữ mang thai / cho con bú:**\n" \
               "  * Tránh dùng **Fluoroquinolones** và **Tetracyclines** để đảm bảo an toàn cho sự phát triển của thai nhi.\n" \
               "- 🧓 **Người cao tuổi (> 65 tuổi):**\n" \
               "  * Thận trọng đặc biệt và cần chỉnh liều khi dùng **Aminoglycosides** (Streptomycin, Gentamicin) do nguy cơ tích lũy độc tính gây suy thận và điếc không hồi phục.\n" \
               "- 💡 **Nguyên tắc chung:** Luôn kết hợp hướng dẫn điều trị quốc gia và kết quả kháng sinh đồ thực tế tại viện."

    if any(k in msg_lower for k in ["dị ứng", "di ung", "penicillin"]):
        return "### ⚠️ Hướng dẫn Lâm sàng cho Bệnh nhân Dị ứng Kháng sinh (Offline Mode)\n\n" \
               "- 💊 **Dị ứng Penicillin:**\n" \
               "  * **Kháng sinh thay thế:** Thường ưu tiên nhóm **Macrolides** (Azithromycin, Clarithromycin, Erythromycin) hoặc Lincosamides (Clindamycin) cho nhiễm khuẩn thông thường.\n" \
               "  * **Lưu ý phản ứng chéo:** Khoảng 3-10% bệnh nhân dị ứng Penicillin có nguy cơ xảy ra phản ứng dị ứng chéo với các kháng sinh nhóm **Cephalosporins** (đặc biệt là Cephalosporin thế hệ 1 như Cephalexin, Cefadroxil). Cần hết sức thận trọng khi kê đơn nhóm này.\n" \
               "- 🩺 **Nguyên tắc chung:** Luôn hỏi rõ tiền sử dị ứng của bệnh nhân (phát ban, ngứa, khó thở hay sốc phản vệ) và ưu tiên thực hiện test da trước khi tiêm/truyền các kháng sinh có nguy cơ cao."

    # 3d. Câu hỏi về mô hình học máy / thuật toán / độ chính xác / dữ liệu (ML & Project FAQ)
    if any(k in msg_lower for k in ["mô hình", "mo hinh", "thuật toán", "thuat toan", "học máy", "hoc may", "độ chính xác", "do chinh xac", "chỉ số", "chi so", "accuracy", "f1", "recall", "auc", "dữ liệu", "du lieu", "mẫu", "stacking", "threshold", "ngưỡng"]):
        return "### 📊 Thông tin chi tiết về Mô hình Học máy & Đồ án (Offline Mode)\n\n" \
               "- 📂 **Thông tin tập dữ liệu (Dataset):**\n" \
               "  * **Số lượng:** **2,404 mẫu** hệ gen vi khuẩn E. coli.\n" \
               "  * **Đặc trưng ban đầu:** 310 đặc trưng (210 đặc trưng gen kháng thuốc AMR và 100 đặc trưng liên tục đại diện cho mật độ k-mer nền).\n" \
               "  * **Đặc trưng rút gọn (sau RFE):** Rút gọn xuống **93 đặc trưng gen/k-mer quan trọng nhất** giúp tối ưu hóa chi phí giải trình tự gen.\n" \
               "- 🤖 **Thuật toán học máy đề xuất:**\n" \
               "  * **XGBoost Pipeline (Đề xuất):** Mô hình phân loại XGBoost được tối ưu hóa siêu tham số bằng Optuna kết hợp với bộ lọc giảm chiều đặc trưng RFE và kỹ thuật cân bằng dữ liệu SMOTE để tối ưu hóa hiệu năng chẩn đoán.\n" \
               "- 📈 **Chỉ số đánh giá mô hình (Performance Metrics):**\n" \
               "  * **Độ chính xác (Accuracy):** **83.00%** trên tập test.\n" \
               "  * **ROC-AUC:** **90.29%** và **PR-AUC:** **89.05%**.\n" \
               "  * **Recall lớp Kháng Ciprofloxacin (Resistant):** Đạt **80.82%** (CV) và **78.00%** (test set) nhờ kỹ thuật cân bằng dữ liệu **SMOTE** (tránh bỏ sót chủng vi khuẩn kháng thuốc trong chẩn đoán vi sinh).\n" \
               "- ⚙️ **Ngưỡng quyết định (Decision Threshold):**\n" \
               "  * Được tối ưu ở mức **0.521** giúp cân bằng hoàn hảo giữa độ chính xác và độ nhạy lâm sàng."

    # 3e. Câu hỏi tình huống lâm sàng chuyên sâu (MDR, MIC, Mang thai + Dị ứng + Kháng parC, Thăt bại Carbapenem)
    if any(k in msg_lower for k in ["đồng thời", "cả hai", "phối hợp gen", "đa kháng", "mdr", "bơm đẩy đi kèm", "bơm đẩy kết hợp", "ước lượng mic", "dải mic", "xét nghiệm bổ sung", "kiểm chứng", "cấy máu", "mang thai bị dị ứng", "thất bại carbapenem", "không giảm sốt", "72h"]):
        return "### 🩺 Tư vấn Lâm sàng nâng cao & Tình huống đặc biệt (Offline Mode)\n\n" \
               "- 🧬 **Đồng xuất hiện gen kháng & Đa kháng thuốc (MDR):**\n" \
               "  * Sự kết hợp đồng thời của gen sinh ESBL (`blaCTX-M-15`) và đột biến đích Fluoroquinolone (`gyrA_S83L`) tạo ra chủng vi khuẩn đa kháng cực kỳ nguy hiểm. Mức độ nguy hại lâm sàng tăng vọt do hầu như tất cả các kháng sinh Cephalosporin thế hệ 3/4 và Quinolone (Ciprofloxacin) thông thường đều mất tác dụng.\n" \
               "  * Nếu xuất hiện thêm cơ chế bơm đẩy chủ động (efflux pump như `floR`, `tet`), vi khuẩn có thể tự động đẩy bớt kháng sinh ra ngoài, làm giảm nồng độ thuốc nội bào và thúc đẩy tính kháng thuốc chéo.\n" \
               "- 🔬 **Nồng độ MIC & Kiểm chứng vi sinh:**\n" \
               "  * Mô hình XGBoost Pipeline hiện tại chỉ phân loại nhị phân Kháng/Nhạy đối với Ciprofloxacin dựa trên kiểu gen. Hệ thống **không** dự đoán trực tiếp giá trị MIC (Nồng độ ức chế tối thiểu) bằng số.\n" \
               "  * Nghiên cứu viên nên chỉ định thêm **Kháng sinh đồ đĩa giấy khuếch tán (Kirby-Bauer)** hoặc máy tự động (như Vitek 2) để xác định chính xác MIC thực tế của chủng vi khuẩn.\n" \
               "- 🤰 **Ca nhiễm E. coli phức tạp (Mang thai + Dị ứng Penicillin + Kháng parC):**\n" \
               "  * Do mẫu kháng Fluoroquinolone (parC đột biến) => Không dùng Ciprofloxacin.\n" \
               "  * Bệnh nhân dị ứng Penicillin => Tránh dùng các penicillin.\n" \
               "  * Bệnh nhân mang thai => Tránh cả Quinolone và Tetracycline.\n" \
               "  * **Giải pháp thay thế khả thi:** Có thể cân nhắc các kháng sinh nhóm **Macrolides** (như Azithromycin) nếu vi khuẩn nhạy cảm, hoặc chuyển sang **Carbapenem** (như Meropenem) trong trường hợp nhiễm trùng nặng đe dọa tính mạng (do Carbapenem tương đối an toàn trong thai kỳ và tỷ lệ dị ứng chéo với Penicillin cực kỳ thấp, dưới 1%).\n" \
               "- 🌡️ **Thất bại điều trị Carbapenem sau 72 giờ:**\n" \
               "  * Cần đánh giá lại xem có ổ nhiễm trùng sâu chưa được dẫn lưu hay không (như áp xe).\n" \
               "  * Kiểm tra xem vi khuẩn có sinh enzyme Carbapenemase (như gen *blaNDM*, *blaKPC* - kháng cả Carbapenem) hay không.\n" \
               "  * Cần hội chẩn chuyên khoa truyền nhiễm để chuyển sang phối hợp thuốc (ví dụ: Colistin phối hợp hoặc thế hệ mới Ceftazidime-Avibactam)."

    # 3f. Câu hỏi về Sinh tin học / k-mer nền / Gram âm - dương / ngoại lai / tet(A) SHAP âm
    if any(k in msg_lower for k in ["k-mer nền", "mật độ k-mer", "tương quan màng", "loài vi khuẩn", "gram dương", "gram âm", "ngoại lai", "anomaly", "gen mới", "ngoài tập train", "ép vào dự đoán", "tet(a)", "shap âm"]):
        return "### 🧬 Giải đáp Sinh tin học & Cơ chế Mô hình Học máy (Offline Mode)\n\n" \
               "- 📊 **Giải thích tín hiệu từ k-mer nền (Background k-mers):**\n" \
               "  * Trong trường hợp không tìm thấy gen kháng cụ thể nhưng mô hình vẫn dự đoán Kháng (Resistant), tín hiệu này đến từ sự thay đổi mật độ của các k-mer ngắn trong hệ gen vi khuẩn E. coli.\n" \
               "  * Sự thay đổi này thường tương quan với các đột biến cấu trúc màng tế bào (làm giảm tính thấm porin màng của vi khuẩn Gram âm như *E. coli*), hoặc phản ánh nguồn gốc tiến hóa của chủng kháng thuốc.\n" \
               "- 📁 **Đối tượng huấn luyện & Giới hạn dữ liệu:**\n" \
               "  * Mô hình được huấn luyện tối ưu nhất trên loài *Escherichia coli* và họ *Enterobacteriaceae*.\n" \
               "  * **Cảnh báo:** Độ tin cậy của mô hình sẽ **giảm mạnh** nếu đưa vào mẫu vi khuẩn Gram dương do sự khác biệt hoàn toàn về cấu trúc vách tế bào và cơ chế kháng thuốc.\n" \
               "  * **anomaly detection (Cảnh báo ngoại lai):** Hiện tại mô hình XGBoost Pipeline không có bộ lọc phát hiện dị thường. Nếu đưa vào một gen hoàn toàn mới hoặc loài vi khuẩn không phù hợp, mô hình vẫn ép đưa ra dự đoán nhị phân đối với Ciprofloxacin.\n" \
               "- 🧬 **Tại sao gen tet(A) có SHAP âm ở một số mẫu cụ thể (ví dụ mẫu ID 1042)?**\n" \
               "  * SHAP đo lường đóng góp tương tác giữa các đặc trưng. Mặc dù `tet(A)` là gen kháng Tetracycline, nhưng nếu trong mẫu đó thiếu các gen đồng tác nhân hoặc mật độ k-mer nền thể hiện kiểu gen của một chủng nhạy cảm yếu, sự đóng góp cục bộ của đặc trưng này có thể bị bù trừ bởi các yếu tố khác, tạo ra giá trị SHAP âm."

    # 3g. Câu hỏi về UX / tính năng web / FASTA / FASTQ / tải biểu đồ / lưu lịch sử
    if any(k in msg_lower for k in ["fasta", "fastq", "trình tự", "tải biểu đồ", "lưu lịch sử", "luu lich su", "tiến triển"]):
        return "### 🤖 Giải đáp về Tính năng và Trải nghiệm Hệ thống (Web UX)\n\n" \
               "- 🧬 **Đầu vào FASTA/FASTQ:**\n" \
               "  * Hiện tại ứng dụng web **chưa hỗ trợ** tải lên file thô FASTA hoặc FASTQ trực tiếp.\n" \
               "  * Người dùng cần chạy quy trình trích xuất đặc trưng sinh học trước (ví dụ: dùng công cụ tìm gen kháng AMR FinderPlus hoặc công cụ đếm k-mer Jellyfish) để tạo ra file CSV dạng bảng trước khi đưa vào web dự đoán.\n" \
               "- 📊 **Tải biểu đồ SHAP dạng ảnh / PDF:**\n" \
               "  * Bạn có thể nhấn chuột phải trực tiếp vào biểu đồ SHAP (được vẽ bằng thư viện Chart.js trên web) và chọn **'Save image as...' (Lưu ảnh dưới dạng...)** để tải về máy dưới dạng ảnh PNG chất lượng cao phục vụ viết báo cáo.\n" \
               "- 💾 **Lưu lịch sử dự đoán chủng vi khuẩn:**\n" \
               "  * Phiên bản hiện tại lưu lịch sử trực tiếp vào cơ sở dữ liệu SQLite (`amr_history.db`) ở phía Back-end."

    # 4. Câu hỏi về SHAP/giải thích mô hình
    if any(k in msg_lower for k in ["shap", "biểu đồ", "đồ thị", "giải thích"]):
        return "### 📊 Giải thích về SHAP (Shapley Additive exPlanations)\n\n" \
               "- **SHAP** đo lường mức độ đóng góp của từng gen / k-mer vào quyết định của mô hình XGBoost Pipeline.\n" \
               "- **Cột màu Đỏ (SHAP > 0):** Đại diện cho các yếu tố thúc đẩy chủng vi khuẩn trở nên **Kháng Ciprofloxacin**.\n" \
               "- **Cột màu Xanh (SHAP < 0):** Đại diện cho các yếu tố giữ chủng vi khuẩn ở trạng thái **Nhạy cảm Ciprofloxacin**.\n" \
               "- Độ dài của cột tỉ lệ thuận với độ mạnh của tác động."

    # 5. Câu hỏi mặc định
    return "Tôi là **Trợ lý Nghiên cứu AMR (Local Expert Mode)**. Tôi chưa thể hiểu hết câu hỏi phức tạp này ở chế độ ngoại tuyến.\n\n" \
           "**Mẹo:** Bạn có thể hỏi về các gen cụ thể (ví dụ: *'blaCTX-M-15 là gì?'*), hỏi về phác đồ điều trị (ví dụ: *'kê đơn kháng sinh gì?'*), hoặc giải thích đồ thị (ví dụ: *'SHAP là gì?'*). Nếu muốn hỏi về mô hình, bạn có thể hỏi *'độ chính xác'* hoặc *'thuật toán là gì'*.\n\n" \
           "*Để được hỗ trợ phân tích hội thoại tự do chuyên sâu bằng mô hình GenAI, vui lòng cấu hình API Key trong file .env.*"

def generate_ai_report(outcome, probability, top_features, threshold):
    """Gọi API của Gemini dựa trên API Key được cấu hình trong file .env (có xoay vòng backup)."""
    api_keys = get_gemini_api_keys()
    gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')

    outcome_vietnamese = "KHÁNG CIPROFLOXACIN (Resistant)" if outcome == "Resistant" else "NHẠY CẢM CIPROFLOXACIN (Susceptible)"
    
    prompt = f"""
    Bạn là một Cố vấn Sinh tin học và Vi sinh lâm sàng chuyên ngành vi khuẩn E. coli và đề kháng kháng sinh.
    Hãy viết một báo cáo phân tích mức độ đề kháng/nhạy cảm kháng sinh Ciprofloxacin ngắn gọn, chuyên nghiệp bằng tiếng Việt cho chủng E. coli này dựa trên kết quả dự đoán kiểu hình sau:
    
    - Dự báo của mô hình XGBoost Pipeline: {outcome_vietnamese} (Xác suất đề kháng Ciprofloxacin: {probability * 100:.2f}%, Ngưỡng quyết định: {threshold:.3f})
    - Top các đặc trưng gen/k-mer ảnh hưởng lớn nhất lấy từ giải thích SHAP:
    {json.dumps(top_features, indent=2)}
    
    Yêu cầu báo cáo:
    1. Tóm tắt mức đề kháng Ciprofloxacin của chủng vi khuẩn E. coli này.
    2. Giải thích ý nghĩa sinh học của các đột biến/gen kháng thuốc chính xuất hiện trong danh sách (đặc biệt là các đột biến điểm vùng QRDR của gyrA/parC có tác động SHAP dương lớn thúc đẩy kết quả kháng Ciprofloxacin).
    3. Đưa ra các gợi ý/khuyến nghị điều trị nhiễm trùng do chủng này gây ra (ví dụ: tránh sử dụng Ciprofloxacin/Fluoroquinolone, đề xuất các kháng sinh thay thế khả thi như Carbapenem hoặc yêu cầu làm thêm kháng sinh đồ kiểm chứng).
    4. Giữ giọng văn khoa học vi sinh chuyên nghiệp, cấu trúc rõ ràng sử dụng Markdown. Không dài dòng lê thê.
    """

    # --- Thử gọi Gemini API bằng danh sách Key khả dụng ---
    for i, key in enumerate(api_keys):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={key}"
        key_label = "Chính" if i == 0 else f"Backup {i}"
        print(f"📡 [AI Report] Đang gọi Gemini API (Model: {gemini_model}) bằng Key {key_label}...")
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                print(f"✅ [AI Report] Gọi Gemini ({gemini_model}) bằng Key {key_label} thành công!")
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"❌ [AI Report] Lỗi gọi Gemini API bằng Key {key_label}: {e}")
            # Tiếp tục vòng lặp sang key tiếp theo

    # Fallback về Local Expert System nếu tất cả API key đều lỗi hoặc trống
    print("⚠️ [AI Report] Tất cả Gemini API Keys đều thất bại. Tự động chuyển về Hệ chuyên gia cục bộ (Local).")
    return f"🧬 **[Local Expert Mode — Chế độ Ngoại tuyến]** *Không có kết nối Gemini API. Hệ thống tự động kích hoạt bộ luật y khoa cục bộ để phân tích đặc trưng gen...*\n\n" + \
           generate_local_report(outcome, probability, top_features, threshold)

def call_ai_chat(system_instruction, history, user_message):
    """Gửi lịch sử hội thoại và system instruction tới Gemini API (có xoay vòng backup)."""
    api_keys = get_gemini_api_keys()
    gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')

    if not api_keys:
        raise ValueError("Không cấu hình API Key nào trong file .env")

    # Chuẩn hóa lịch sử chat cho Gemini (role 'user' và 'model')
    contents = []
    for msg in history:
        role = msg.get("role", "user")
        role = "user" if role == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": contents
    }
    headers = {"Content-Type": "application/json"}

    # --- Thử gọi Gemini API bằng danh sách Key khả dụng ---
    last_err = None
    for i, key in enumerate(api_keys):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={key}"
        key_label = "Chính" if i == 0 else f"Backup {i}"
        print(f"💬 [AI Chat] Đang gọi Gemini API (Model: {gemini_model}) bằng Key {key_label}...")
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                print(f"✅ [AI Chat] Gọi Gemini ({gemini_model}) bằng Key {key_label} thành công!")
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"❌ [AI Chat] Lỗi gọi Gemini Chat API bằng Key {key_label}: {e}")
            last_err = e
            # Tiếp tục vòng lặp sang key tiếp theo

    raise last_err if last_err else ValueError("Tất cả API keys đều lỗi.")

app = Flask(__name__)

# Cấu hình đường dẫn thư mục tải lên tạm thời
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Biến toàn cục để lưu mô hình và các thông tin liên quan
MODEL = None
THRESHOLD = None
FEATURES = None
SHAP_EXPLAINER = None
MODEL_PATH = None

def init_model():
    """Tự động tìm và load mô hình mới nhất trong thư mục."""
    global MODEL, THRESHOLD, FEATURES, SHAP_EXPLAINER, MODEL_PATH
    
    # Khởi tạo DB SQLite trước khi nạp mô hình
    init_db()
    
    model_files = glob.glob("models/amr_classifier_*.joblib")
    if not model_files:
        print("Warning: No .joblib model files found. Please run run_training.py first.")
        return False
    
    # Lấy file mô hình mới nhất theo thứ tự bảng chữ cái (chứa timestamp)
    MODEL_PATH = sorted(model_files)[-1]
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        MODEL, THRESHOLD, FEATURES = ml_pipeline.load_model(MODEL_PATH)
        print("Success: Model loaded successfully!")
        
        # Tải dữ liệu nền (background) để khởi tạo SHAP Explainer
        # Nếu có file CSV dữ liệu, dùng 15 mẫu đầu tiên làm background (giảm từ 100 để tăng tốc độ trên Render)
        if os.path.exists("data/X.csv"):
            X_background = pd.read_csv("data/X.csv", index_col=0).head(15)
            print("Initializing SHAP Explainer...")
            SHAP_EXPLAINER = ml_pipeline.build_shap_explainer(MODEL, X_background)
            print("Success: SHAP Explainer initialized successfully!")
        else:
            print("Warning: X.csv not found. SHAP features will be disabled.")
        return True
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        return False

# ----------------- WEB PAGES -----------------

@app.route('/')
def home():
    """Trang chủ hiển thị Giao diện Dashboard."""
    return render_template('index.html')

# ----------------- API ENDPOINTS -----------------

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Trả về thông tin chi tiết của mô hình hiện tại."""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded yet."}), 500
    
    return jsonify({
        "status": "success",
        "model_name": os.path.basename(MODEL_PATH),
        "threshold": round(THRESHOLD, 3),
        "features_count": len(FEATURES),
        "shap_enabled": SHAP_EXPLAINER is not None
    })

@app.route('/api/gene_db', methods=['GET'])
def get_gene_db():
    """Trả về cơ sở dữ liệu luật gen kháng thuốc để hiển thị từ điển trên frontend."""
    return jsonify({
        "status": "success",
        "gene_db": GENE_DB
    })

@app.route('/api/get_samples', methods=['GET'])
def get_samples():
    """Trả về danh sách các mẫu chủng vi khuẩn đại diện nhạy cảm và kháng thuốc được phân loại chính xác."""
    if not os.path.exists("data/X.csv") or not os.path.exists("data/y.csv"):
        return jsonify({"status": "error", "message": "Data files not found."}), 404
    
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded yet."}), 500
        
    try:
        X = pd.read_csv("data/X.csv", index_col=0)
        y = pd.read_csv("data/y.csv", index_col=0).iloc[:, 0]
        
        # Dự đoán xác suất cho toàn bộ dữ liệu để lọc mẫu chính xác
        df_aligned = X.reindex(columns=FEATURES, fill_value=0)
        probabilities = MODEL.predict_proba(df_aligned)[:, 1]
        
        df_helper = pd.DataFrame({
            "true_label": y,
            "prob": probabilities
        }, index=X.index)
        
        # Lọc mẫu Nhạy cảm chuẩn (y_true == 0 và y_prob < 0.2)
        sus_candidates = df_helper[(df_helper["true_label"] == 0) & (df_helper["prob"] < 0.2)]
        if sus_candidates.empty:
            sus_candidates = df_helper[(df_helper["true_label"] == 0) & (df_helper["prob"] < THRESHOLD)]
            
        # Lọc mẫu Kháng chuẩn (y_true == 1 và y_prob > 0.8)
        res_candidates = df_helper[(df_helper["true_label"] == 1) & (df_helper["prob"] > 0.8)]
        if res_candidates.empty:
            res_candidates = df_helper[(df_helper["true_label"] == 1) & (df_helper["prob"] >= THRESHOLD)]
            
        # Chọn ngẫu nhiên hoặc lấy mẫu tối đa 5 phần tử đại diện
        import random
        
        sus_samples = []
        res_samples = []
        
        # Lấy tối đa 5 mẫu nhạy cảm
        sus_ids = list(sus_candidates.index)
        selected_sus_ids = random.sample(sus_ids, min(5, len(sus_ids))) if sus_ids else []
        for sid in selected_sus_ids:
            sus_samples.append({
                "id": str(sid),
                "features": X.loc[sid].to_dict(),
                "true_label": "Susceptible",
                "prob": float(df_helper.loc[sid, "prob"])
            })
            
        # Lấy tối đa 5 mẫu kháng thuốc
        res_ids = list(res_candidates.index)
        selected_res_ids = random.sample(res_ids, min(5, len(res_ids))) if res_ids else []
        for rid in selected_res_ids:
            res_samples.append({
                "id": str(rid),
                "features": X.loc[rid].to_dict(),
                "true_label": "Resistant",
                "prob": float(df_helper.loc[rid, "prob"])
            })
            
        return jsonify({
            "status": "success",
            "samples": {
                "susceptible": sus_samples,
                "resistant": res_samples
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API dự đoán cho 1 chủng E. coli cụ thể.
    Yêu cầu JSON body: { "features": { "gene1": 0, "gene2": 1, ... }, "strain_id": "EC..." }
    """
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500
    
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"status": "error", "message": "Missing 'features' in request body."}), 400
    
    try:
        # Chuyển đổi dữ liệu gửi lên thành Pandas Series
        feature_dict = data['features']
        feature_vector = pd.Series(feature_dict)
        
        # Nhận diện mã chủng vi khuẩn, tự động sinh nếu trống
        strain_id = data.get('strain_id', data.get('patient_id', '')).strip()
        if not strain_id:
            import datetime
            import random
            now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
            rand_num = random.randint(100, 999)
            strain_id = f"EC-{now_str}-{rand_num}"
        
        # 1. Dự đoán kết quả
        prediction_res = ml_pipeline.predict_one_patient(feature_vector, MODEL, THRESHOLD, FEATURES)
        
        # 2. Giải thích SHAP nếu khả dụng
        shap_explanation = None
        if SHAP_EXPLAINER is not None:
            # Lấy top 10 đặc trưng ảnh hưởng nhiều nhất
            shap_res = ml_pipeline.explain_prediction(SHAP_EXPLAINER, feature_vector, FEATURES, top_k=10)
            shap_explanation = shap_res
            
        # 3. Tạo báo cáo sinh tin học AI (Gemini hoặc Local Expert System fallback)
        top_features = shap_explanation['top_features'] if shap_explanation else []
        ai_report = generate_ai_report(
            prediction_res['prediction'], 
            prediction_res['prob_resistant'], 
            top_features,
            THRESHOLD
        )
        
        # 4. Lưu vào cơ sở dữ liệu SQLite
        try:
            detected_genes_list = []
            for f in top_features:
                name = f['feature']
                val = f['feature_value']
                if name in GENE_DB and val > 0:
                    detected_genes_list.append(name)
            detected_genes_str = ", ".join(detected_genes_list) if detected_genes_list else "Không phát hiện"
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_history (strain_id, prediction, probability, detected_genes, features_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                strain_id,
                prediction_res['prediction'],
                prediction_res['prob_resistant'],
                detected_genes_str,
                json.dumps(feature_dict)
            ))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Error saving prediction to database: {db_err}")
            
        return jsonify({
            "status": "success",
            "prediction": prediction_res,
            "shap": shap_explanation,
            "ai_report": ai_report,
            "strain_id": strain_id,
            "patient_id": strain_id  # Khóa tương thích ngược cho frontend cũ
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    API dự đoán hàng loạt từ file CSV tải lên.
    Trả về kết quả dưới dạng JSON (để hiển thị bảng) và thông tin file.
    """
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
    
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Đọc CSV
            df = pd.read_csv(file_path, index_col=0)
            
            # Đảm bảo các cột khớp với đặc trưng khi train
            # Những cột thiếu sẽ được gán giá trị 0
            df_aligned = df.reindex(columns=FEATURES, fill_value=0)
            
            # Dự đoán xác suất
            probabilities = MODEL.predict_proba(df_aligned)[:, 1]
            predictions = ["Resistant" if p >= THRESHOLD else "Susceptible" for p in probabilities]
            
            # Lưu kết quả dự đoán vào DataFrame để cho phép tải về
            result_df = pd.DataFrame({
                "Sample_ID": df.index,
                "Prediction": predictions,
                "Probability_Resistant": np.round(probabilities, 4)
            })
            
            output_filename = f"predicted_{file.filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            result_df.to_csv(output_path, index=False)
            
            # Trả về tối đa 50 dòng kết quả đầu tiên để vẽ bảng trên web
            preview_data = result_df.head(50).to_dict(orient='records')
            
            return jsonify({
                "status": "success",
                "total_records": len(result_df),
                "download_url": f"/download/{output_filename}",
                "preview": preview_data
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            # Xóa file upload tạm để tránh rác hệ thống
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"status": "error", "message": "Invalid file format. Only CSV is allowed."}), 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Đường dẫn tải xuống file kết quả dự đoán hàng loạt."""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API gửi tin nhắn hội thoại sinh tin học với Gemini hoặc hệ chuyên gia cục bộ.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"status": "error", "message": "Missing 'message' in request body."}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    prediction_context = data.get('context', {})
    
    api_keys = get_gemini_api_keys()
    
    outcome = prediction_context.get('prediction', 'Unknown')
    prob = prediction_context.get('prob_resistant', 0.0)
    top_features = prediction_context.get('top_features', [])
    
    outcome_vietnamese = "KHÁNG CIPROFLOXACIN (Resistant)" if outcome == "Resistant" else "NHẠY CẢM CIPROFLOXACIN (Susceptible)"
    system_instruction = f"""
    Bạn là một Cố vấn Sinh tin học và Vi sinh lâm sàng chuyên ngành vi khuẩn E. coli và đề kháng kháng sinh.
    Bạn đang trao đổi với kỹ thuật viên vi sinh hoặc bác sĩ điều trị về đặc tính kháng thuốc Ciprofloxacin của chủng vi khuẩn E. coli có kết quả như sau:
    - Dự báo của mô hình XGBoost Pipeline: {outcome_vietnamese} (Xác suất đề kháng Ciprofloxacin: {prob * 100:.2f}%)
    - Top đặc trưng ảnh hưởng lớn nhất lấy từ giải thích SHAP: {json.dumps(top_features)}
    
    Hãy trả lời các câu hỏi một cách khoa học, chuyên nghiệp, súc tích và dựa trên nghiên cứu vi sinh lâm sàng.
    Trả lời bằng tiếng Việt, cấu trúc rõ ràng bằng markdown.
    Nếu họ hỏi về phác đồ điều trị cụ thể, hãy gợi ý hướng sử dụng (tránh Fluoroquinolone/Ciprofloxacin nếu đề kháng, cân nhắc nhóm Carbapenem hoặc nhóm thay thế khác) và nhắc nhở họ rằng đây là tư vấn hỗ trợ và họ cần tham chiếu hướng dẫn điều trị quốc gia và kết quả kháng sinh đồ thực tế.
    """
    
    # Kiểm tra xem có bất kỳ API key Gemini nào được cấu hình hay không
    if not api_keys:
        reply = generate_local_chat_reply(user_message, outcome, prob, top_features)
        return jsonify({"status": "success", "reply": reply})
        
    try:
        reply = call_ai_chat(system_instruction, history, user_message)
        return jsonify({"status": "success", "reply": reply})
    except Exception as e:
        print(f"Chat API error fallback to local: {e}")
        reply = f"🤖 **[Local Advisor — Chế độ Ngoại tuyến]** *Không thể kết nối dịch vụ trực tuyến. Đang tự động trả lời bằng dữ liệu luật kháng thuốc cục bộ...*\n\n" + \
                generate_local_chat_reply(user_message, outcome, prob, top_features)
        return jsonify({"status": "success", "reply": reply})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Lấy danh sách lịch sử dự đoán từ cơ sở dữ liệu SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, strftime('%Y-%m-%d %H:%M:%S', datetime(timestamp, 'localtime')) as formatted_time, 
                   strain_id, prediction, probability, detected_genes, features_json
            FROM prediction_history 
            ORDER BY timestamp DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for r in rows:
            history_list.append({
                "id": r["id"],
                "timestamp": r["formatted_time"],
                "strain_id": r["strain_id"],
                "patient_id": r["strain_id"],  # Khóa tương thích ngược cho frontend cũ
                "prediction": r["prediction"],
                "probability": r["probability"],
                "detected_genes": r["detected_genes"] if r["detected_genes"] else "Không phát hiện",
                "features": json.loads(r["features_json"])
            })
            
        return jsonify({
            "status": "success",
            "history": history_list
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history/delete', methods=['POST'])
def delete_history_item():
    """Xóa một bản ghi lịch sử theo ID."""
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({"status": "error", "message": "Missing 'id' in request body."}), 400
    
    entry_id = data['id']
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history WHERE id = ?', (entry_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Deleted entry {entry_id} successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_all_history():
    """Xóa toàn bộ lịch sử dự đoán."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history')
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Cleared all prediction history successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/epidemiology_stats', methods=['GET'])
def get_epidemiology_stats():
    """Thống kê dữ liệu dịch tễ học từ cơ sở dữ liệu SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Thống kê tỷ lệ kháng thuốc theo ngày
        cursor.execute('''
            SELECT date(timestamp) as day_str,
                   count(*) as total,
                   sum(case when prediction = 'Resistant' then 1 else 0 end) as resistant_count
            FROM prediction_history
            GROUP BY day_str
            ORDER BY day_str ASC
        ''')
        rows_by_day = cursor.fetchall()
        
        timeline_data = []
        for r in rows_by_day:
            day = r["day_str"]
            total = r["total"]
            resistant = r["resistant_count"]
            rate = round((resistant / total) * 100, 1) if total > 0 else 0.0
            timeline_data.append({
                "date": day,
                "total": total,
                "resistant": resistant,
                "rate": rate
            })
            
        # 2. Thống kê tần suất xuất hiện của các gen kháng thuốc
        cursor.execute('SELECT detected_genes FROM prediction_history')
        all_genes_rows = cursor.fetchall()
        
        gene_counts = {}
        for row in all_genes_rows:
            genes_str = row["detected_genes"]
            if genes_str and genes_str != "Không phát hiện":
                genes = [g.strip() for g in genes_str.split(",")]
                for g in genes:
                    if g:
                        gene_counts[g] = gene_counts.get(g, 0) + 1
                        
        sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
        gene_stats = [{"gene": g, "count": c} for g, c in sorted_genes]
        
        conn.close()
        
        return jsonify({
            "status": "success",
            "timeline": timeline_data,
            "genes": gene_stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------------- SETUP AND RUN -----------------

# Khởi tạo mô hình ngay khi server start
init_model()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
