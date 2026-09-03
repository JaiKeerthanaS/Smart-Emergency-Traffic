import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path

from utils.detector import Detector
from utils.video_processor import VideoProcessor
from utils.signal_controller import SignalController

# Configuration
VIDEO_DIR = Path("videos")
MODEL_PATH = Path("models/best.pt")

GREEN_TIME = 8
YELLOW_TIME = 2
INFERENCE_INTERVAL = 5  # run YOLO every N frames
DETECTION_CONFIRM = 3  # frames required to confirm
DETECTION_CLEAR = 5  # frames required to clear

st.set_page_config(page_title="Smart Emergency Vehicle Priority System", layout="wide")

def init_session():
    if "video_processors" not in st.session_state:
        st.session_state.video_processors = {}

    if "detector" not in st.session_state:
        st.session_state.detector = Detector(MODEL_PATH, confidence=0.5)

    if "signal_controller" not in st.session_state:
        st.session_state.signal_controller = SignalController(
            green=GREEN_TIME,
            yellow=YELLOW_TIME,
            confirm_frames=DETECTION_CONFIRM,
            clear_frames=DETECTION_CLEAR
        )

    if "events" not in st.session_state:
        st.session_state.events = []

    if "frame_counts" not in st.session_state:
        st.session_state.frame_counts = [0, 0, 0, 0]

    # Store the most recent YOLO detection for each road
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = [None, None, None, None]

    if "prototype_mode" not in st.session_state:
        st.session_state.prototype_mode = "DEMO"

    if "demo_selection" not in st.session_state:
        st.session_state.demo_selection = {
            "road": None,
            "vehicle": None,
            "active": False
        }


def add_event(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.events.insert(0, f"{ts} — {msg}")
    # keep only latest 20
    st.session_state.events = st.session_state.events[:20]


def camera_panel(container, vp: VideoProcessor, idx: int, signal_state: str, detection_info: dict | None):
    with container:
        st.markdown(f"**ROAD {idx+1} / CAMERA {idx+1}**")
        if not vp.available:
            st.warning(f"Camera feed unavailable — add road{idx+1}.mp4 to the videos folder.")
            st.text("Camera feed unavailable — add videos/road{}.mp4 to the videos folder.".format(idx+1))
            return

        frame = vp.read_frame()
        if frame is None:
            st.info("Video ended or unreadable — looping or unavailable.")
            st.empty()
            return

        # draw detection if any
        if detection_info:
            for det in detection_info.get("boxes", []):
                x1, y1, x2, y2 = map(int, det["box"])
                label = f"{det['label']} {det['conf']*100:.0f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, use_container_width=True)

        st.write(f"Signal: {signal_state}")
        if detection_info and detection_info.get("detected"):
            st.success(f"Detected: {detection_info.get('vehicle_type').upper()} ({detection_info.get('confidence')*100:.1f}%)")


def main():
    init_session()

    st.title("🚨 Smart Emergency Vehicle Priority System")
    st.subheader("AI-Based Real-Time Traffic Signal Management")

    # Sidebar
    with st.sidebar:
        st.markdown("**Prototype Mode**")
        mode = st.radio("Mode", ["AI MODE", "DEMO MODE"], index=0 if st.session_state.prototype_mode=="AI" else 1)
        st.session_state.prototype_mode = "AI" if mode == "AI MODE" else "DEMO"

        st.markdown("---")
        st.markdown("**DEMO Controls**")
        sim_vehicle = st.selectbox("Vehicle", ["Ambulance", "Fire Truck", "Police"])
        sim_road = st.selectbox("Road", ["Road 1", "Road 2", "Road 3", "Road 4"])
        if st.button("🚨 ACTIVATE EMERGENCY"):
            st.session_state.demo_selection.update({"road": sim_road, "vehicle": sim_vehicle, "active": True})
            add_event(f"DEMO SIMULATION ACTIVATED — {sim_vehicle} on {sim_road}")
            st.session_state.signal_controller.activate_emergency(sim_road, sim_vehicle, 0.99, simulated=True)

        if st.button("Stop DEMO"):
            st.session_state.demo_selection.update({"road": None, "vehicle": None, "active": False})
            st.session_state.signal_controller.clear_emergency()
            add_event("DEMO SIMULATION STOPPED")

        st.markdown("---")
        st.markdown("**System Info**")
        st.write("Prototype / Simulation — Not connected to real traffic infrastructure.")

    # Initialize video processors for four roads
    vps = []
    for i in range(4):
        path = VIDEO_DIR / f"road{i+1}.mp4"
        key = str(path)
        if key not in st.session_state.video_processors:
            st.session_state.video_processors[key] = VideoProcessor(path)
        vps.append(st.session_state.video_processors[key])

    # Top status
    ai_status = "LOADED" if st.session_state.detector.model_loaded else "AI MODEL NOT LOADED — DEMO MODE"
    active_cams = sum(1 for vp in vps if vp.available)

    status_col1, status_col2, status_col3, status_col4 = st.columns([1,1,1,1])
    status_ph1 = status_col1.empty()
    status_ph2 = status_col2.empty()
    status_ph3 = status_col3.empty()
    status_ph4 = status_col4.empty()
    status_ph1.metric("System Mode", "EMERGENCY PRIORITY" if st.session_state.signal_controller.is_emergency else ("DEMO MODE" if st.session_state.prototype_mode=="DEMO" else "NORMAL"))
    status_ph2.metric("AI Model Status", ai_status)
    status_ph3.metric("Active Cameras", active_cams)
    status_ph4.metric("Emergency", "YES" if st.session_state.signal_controller.is_emergency else "NO")

    st.markdown("---")

    # Layout for four camera feeds: prepare placeholders and static labels
    cam_cols = [st.columns(2), st.columns(2)]
    cam_items = []  # list of dicts: {image_ph, signal_ph, detect_ph, vp}
    for idx, vp in enumerate(vps):
        colset = cam_cols[0] if idx < 2 else cam_cols[1]
        container = colset[idx % 2].container()
        with container:
            st.markdown(f"**ROAD {idx+1} / CAMERA {idx+1}**")
            image_ph = st.empty()
            signal_ph = st.empty()
            detect_ph = st.empty()
        cam_items.append({"image_ph": image_ph, "signal_ph": signal_ph, "detect_ph": detect_ph, "vp": vp, "idx": idx})

    # Continuous playback loop: read frames, run detection at intervals, update placeholders
    try:
        while True:
            # update top status metrics
            status_ph1.metric("System Mode", "EMERGENCY PRIORITY" if st.session_state.signal_controller.is_emergency else ("DEMO MODE" if st.session_state.prototype_mode=="DEMO" else "NORMAL"))
            status_ph4.metric("Emergency", "YES" if st.session_state.signal_controller.is_emergency else "NO")

            for item in cam_items:
                idx = item["idx"]
                vp = item["vp"]
                image_ph = item["image_ph"]
                signal_ph = item["signal_ph"]
                detect_ph = item["detect_ph"]

                frame = vp.read_frame()
                if frame is None:
                    # show placeholder text
                    image_ph.text("Camera feed unavailable or unreadable.")
                    continue

                # increment frame counter and run YOLO periodically
                st.session_state.frame_counts[idx] += 1

                run_infer = (
                    st.session_state.frame_counts[idx] % INFERENCE_INTERVAL == 0
                )

                detection = None
                if run_infer and st.session_state.prototype_mode == "AI" and st.session_state.detector.model_loaded:
                    try:
                        results = st.session_state.detector.detect_on_frame(frame)

                        emergency = [
                            r for r in results
                            if r["label_norm"] in ("ambulance", "fire truck", "police")
                            and r["conf"] >= st.session_state.detector.confidence
                        ]

                        if emergency:
                            best = max(emergency, key=lambda x: x["conf"])

                            detection = {
                                "detected": True,
                                "vehicle_type": best["label_norm"],
                                "confidence": best["conf"],
                                "boxes": [{
                                    "box": best["box"],
                                    "label": best["label"],
                                    "conf": best["conf"]
                                }]
                            }

                            # Save latest detection for this road
                            st.session_state.last_detections[idx] = detection

                            # Update traffic signal
                            st.session_state.signal_controller.process_detection(
                                idx + 1,
                                detection
                            )

                            add_event(
                                f"AI detected {best['label_norm'].upper()} "
                                f"on Road {idx+1} ({best['conf']*100:.0f}%)"
                            )

                        else:
                            st.session_state.last_detections[idx] = None
                            st.session_state.signal_controller.process_no_detection(idx + 1)

                    except Exception as e:
                        detect_ph.error(f"Detection error: {e}")
                

                # DEMO handling
                if st.session_state.prototype_mode == "DEMO" and st.session_state.demo_selection.get("active"):
                    sel = st.session_state.demo_selection
                    if sel.get("road") == f"Road {idx+1}":
                        detection = {"detected": True, "vehicle_type": sel.get("vehicle"), "confidence": 0.99, "boxes": []}
                        st.session_state.signal_controller.activate_emergency(f"Road {idx+1}", sel.get("vehicle"), 0.99, simulated=True)

                # draw detections on frame
                if detection and detection.get("boxes"):
                    for det in detection.get("boxes", []):
                        x1, y1, x2, y2 = map(int, det["box"])
                        label = f"{det['label']} {det['conf']*100:.0f}%"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # convert and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_ph.image(frame_rgb, use_container_width=True)

                # update signal and detection text
                signal_state = st.session_state.signal_controller.get_signal_state(f"Road {idx+1}")
                signal_ph.write(f"Signal: {signal_state}")
                if detection and detection.get("detected"):
                    detect_ph.success(f"Detected: {detection.get('vehicle_type').upper()} ({detection.get('confidence')*100:.1f}%)")
                else:
                    detect_ph.info("No detection")

            # small sleep to control playback speed
            time.sleep(0.06)
    except Exception:
        # in case Streamlit stops the script, allow graceful exit
        pass

    # Detection panel and event log
    st.markdown("---")
    det_col, log_col = st.columns([1,1])
    with det_col:
        st.header("AI DETECTION")
        if st.session_state.signal_controller.is_emergency:
            info = st.session_state.signal_controller.current_emergency
            st.error("🚨 EMERGENCY DETECTED")
            st.write(f"**Vehicle:** {info.get('vehicle').upper()}")
            st.write(f"**Road:** {info.get('road')}" )
            st.write(f"**Confidence:** {info.get('confidence')*100:.1f}%")
            st.write("**Priority:** HIGH")
            st.write("**Action:** GREEN PRIORITY ENABLED")
        else:
            st.success("🟢 NO EMERGENCY")
            st.write("No emergency vehicle detected.")

    with log_col:
        st.header("Event Log")
        for e in st.session_state.events:
            st.write(e)

    st.markdown("---")
    st.caption("Prototype / Simulation — Not connected to real traffic infrastructure.")


if __name__ == "__main__":
    main()
