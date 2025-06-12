import streamlit as st
import cv2
import numpy as np
from streamlit.components.v1 import html

# Bootstrap styling
bootstrap_css = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
"""

# Inject Bootstrap CSS
html(bootstrap_css)

# App title
st.markdown("""
<div class="container-fluid bg-primary text-white p-3 mb-4">
    <h1 class="text-center">Sistem Deteksi Supir Mengantuk</h1>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'status' not in st.session_state:
    st.session_state.status = "Status: Menunggu"

# Main container
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header bg-primary text-white">
                Kamera Supir
            </div>
            <div class="card-body">
        """, unsafe_allow_html=True)
        
        # Video placeholder
        video_placeholder = st.empty()
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        st.markdown("""
        <div class="d-grid gap-2 d-md-flex justify-content-md-center mt-3">
        """, unsafe_allow_html=True)
        
        start_btn, stop_btn = st.columns(2)
        with start_btn:
            if st.button("Start Kamera", type="primary"):
                st.session_state.camera_active = True
        with stop_btn:
            if st.button("Stop Kamera", type="secondary"):
                st.session_state.camera_active = False
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header bg-primary text-white">
                Status Deteksi
            </div>
            <div class="card-body">
                <div id="status-text" class="alert alert-info">
                    Status: Menunggu
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Camera capture function
def capture_frame():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.frame = frame
            # Simulate detection status (replace with your actual detection logic)
            st.session_state.status = "Status: Normal" if np.random.random() > 0.3 else "Status: Mengantuk!"
    cap.release()

# Main loop
while st.session_state.camera_active:
    capture_frame()
    video_placeholder.image(st.session_state.frame, channels="RGB")
    html(f"""
    <script>
        document.getElementById("status-text").innerHTML = "{st.session_state.status}";
        document.getElementById("status-text").className = "alert { 'alert-danger' if st.session_state.status.includes('Mengantuk') else 'alert-success' }";
    </script>
    """)
    time.sleep(0.1)  # Adjust frame rate

# Show stopped state
if not st.session_state.camera_active and st.session_state.frame is not None:
    video_placeholder.image(st.session_state.frame, channels="RGB")