import streamlit as st
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tempfile

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="NTA Web App Pro", layout="wide", page_icon="🔬")
st.title("🔬 Ứng Dụng NTA: Phân Tích & Theo Dõi Hạt Nano")
st.markdown("Phần mềm mô phỏng thuật toán **Kalman Filter** sửa lỗi nhận diện AI trong môi trường vi lưu chất.")

# ==========================================
# CÁC HẰNG SỐ VẬT LÝ
# ==========================================
PIXEL_TO_NM = 50.0       
FPS = 30.0               
DT = 1.0 / FPS           
KB = 1.380649e-23        
TEMP_K = 298.15          
VISCOSITY = 0.89e-3      

# ==========================================
# LỚP TOÁN HỌC KALMAN FILTER
# ==========================================
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0 
        self.X = np.zeros((4, 1))
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 5.0
        
    def predict(self):
        self.X = np.dot(self.A, self.X)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return float(self.X[0, 0]), float(self.X[1, 0])
        
    def update(self, z):
        Z = np.array([[z[0]], [z[1]]])
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        Y = Z - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, Y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# ==========================================
# GIAO DIỆN ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cài đặt Hệ thống")
uploaded_video = st.sidebar.file_uploader("1. Tải video NTA (mp4)", type=['mp4'])
threshold_val = st.sidebar.slider("2. Độ nhạy sáng (Threshold)", 50, 250, 150)
sensor_error = st.sidebar.slider("3. Tỷ lệ lỗi AI giả lập (%)", 0, 80, 30)

if uploaded_video is not None:
    # Lưu video tạm thời để OpenCV có thể đọc
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    if st.sidebar.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
        cap = cv2.VideoCapture(video_path)
        
        # Tạo khung chứa video trên Web
        video_placeholder = st.empty()
        
        # Các cột hiển thị thông số Real-time
        col1, col2, col3 = st.columns(3)
        status_text = col1.empty()
        size_text = col2.empty()
        coord_text = col3.empty()
        
        # --- Khởi tạo Auto-Lock ---
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            st.error("Không tìm thấy hạt nào! Hãy giảm độ nhạy sáng xuống.")
            st.stop()
            
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        init_x = float(M["m10"] / M["m00"])
        init_y = float(M["m01"] / M["m00"])

        kf = KalmanFilter()
        kf.X[0, 0] = init_x
        kf.X[1, 0] = init_y
        
        trajectory_kalman = [(init_x, init_y)]
        trajectory_measured = [(init_x, init_y)]
        frames, size_array, msd_array = [0], [0], [0]
        
        frame_count = 0
        start_x, start_y = init_x, init_y
        
        # --- Vòng lặp Xử lý Video ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pred_x, pred_y = kf.predict()
            best_match_x, best_match_y = None, None
            min_dist = 40 
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 2: 
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        dist = math.hypot(cx - pred_x, cy - pred_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_match_x, best_match_y = cx, cy

            tracking_status = "LOCKED"
            status_color = (0, 255, 0)
            
            # Mô phỏng lỗi
            if best_match_x is not None and random.random() > (sensor_error / 100.0):
                kf.update((best_match_x, best_match_y))
                trajectory_measured.append((best_match_x, best_match_y))
            else:
                trajectory_measured.append((np.nan, np.nan)) 
                tracking_status = "PREDICTING (AI LOST)"
                status_color = (255, 165, 0)

            final_x, final_y = float(kf.X[0, 0]), float(kf.X[1, 0])
            trajectory_kalman.append((final_x, final_y))
            frames.append(frame_count)
            
            # --- Tính Vật Lý ---
            sd_meters = ((final_x - start_x)**2 + (final_y - start_y)**2) * (PIXEL_TO_NM * 1e-9)**2
            time_elapsed = frame_count * DT
            msd = sd_meters / time_elapsed 
            msd_array.append(msd)
            
            D = msd / 4.0 if msd > 0 else 1e-15
            d_h_meters = (KB * TEMP_K) / (3.0 * math.pi * VISCOSITY * D)
            d_h_nm = min(max(d_h_meters * 1e9, 10), 1000) 
            size_array.append(d_h_nm)

            # --- Vẽ UI ---
            tail_length = 30
            recent_traj = trajectory_kalman[-tail_length:]
            for i in range(1, len(recent_traj)):
                thickness = int(max(1, 4 * (i / tail_length)))
                pt1 = (int(recent_traj[i-1][0]), int(recent_traj[i-1][1]))
                pt2 = (int(recent_traj[i][0]), int(recent_traj[i][1]))
                cv2.line(frame, pt1, pt2, status_color, thickness)

            draw_x, draw_y = int(final_x), int(final_y)
            length = 15
            cv2.line(frame, (draw_x - length, draw_y), (draw_x + length, draw_y), status_color, 2)
            cv2.line(frame, (draw_x, draw_y - length), (draw_x, draw_y + length), status_color, 2)
            cv2.circle(frame, (draw_x, draw_y), 8, status_color, 1)

            # Cập nhật lên Web
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            status_text.metric("Trạng thái", tracking_status)
            size_text.metric("Kích thước hạt (Ước tính)", f"{d_h_nm:.1f} nm")
            coord_text.metric("Tọa độ hiện tại", f"X: {draw_x} | Y: {draw_y}")

        cap.release()
        st.success("✅ Đã hoàn thành Tracking! Đang vẽ đồ thị báo cáo...")

        # ==========================================
        # XUẤT ĐỒ THỊ BÁO CÁO LÊN WEB
        # ==========================================
        st.markdown("---")
        st.subheader("📊 Báo cáo Khoa học (Physics Report)")
        
        meas_x = [p[0] for p in trajectory_measured]
        kal_x = [p[0] for p in trajectory_kalman]
        time_sec = [f * DT for f in frames]

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].plot(frames, meas_x, 'ro', label='AI Sensor (Lỗi)', alpha=0.4)
        axs[0].plot(frames, kal_x, 'g-', label='Kalman Filter (Đã sửa)', linewidth=2)
        axs[0].set_title("Nội suy và sửa lỗi (Trục X)")
        axs[0].set_xlabel("Frames")
        axs[0].set_ylabel("Tọa độ X")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time_sec[10:], size_array[10:], 'm-', linewidth=2)
        final_size = np.mean(size_array[-30:]) if len(size_array) > 30 else size_array[-1]
        axs[1].axhline(y=final_size, color='k', linestyle='--', label=f'Kết quả cuối: {final_size:.1f} nm')
        axs[1].set_title("Ước lượng kích thước hạt qua thời gian")
        axs[1].set_xlabel("Thời gian (giây)")
        axs[1].set_ylabel("Kích thước (nm)")
        axs[1].legend()
        axs[1].grid(True)

        st.pyplot(fig)
else:
    st.info("👈 Hãy tải một đoạn video lên từ menu bên trái để bắt đầu.")