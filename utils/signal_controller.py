import time
from typing import Optional


class SignalController:
    def __init__(self, green: int = 8, yellow: int = 2, confirm_frames: int = 3, clear_frames: int = 5):
        self.green = green
        self.yellow = yellow
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames

        self.current_index = 1  # 1-based road index for normal cycle
        self.cycle_start = time.time()
        self.phase = "GREEN"  # GREEN, YELLOW

        self.is_emergency = False
        self.current_emergency = None
        self._confirm_counts = {1:0,2:0,3:0,4:0}
        self._clear_counts = {1:0,2:0,3:0,4:0}

    def get_signal_state(self, road_name: str) -> str:
        # If emergency active, that road is GREEN, others RED
        if self.is_emergency:
            if self.current_emergency and self.current_emergency.get("road") == road_name:
                return "🟢 GREEN"
            else:
                return "🔴 RED"

        # Normal cycle based on time
        now = time.time()
        elapsed = now - self.cycle_start
        cycle_len = self.green + self.yellow
        # determine which road in cycle by dividing elapsed by cycle_len
        cycle_pos = int(elapsed // cycle_len) % 4 + 1
        phase_time = elapsed % cycle_len
        if phase_time < self.green:
            phase = "GREEN"
        else:
            phase = "YELLOW"

        if int(road_name.split()[-1]) == cycle_pos:
            return f"🟢 {phase}" if phase == "GREEN" else f"🟡 {phase}"
        else:
            return "🔴 RED"

    def process_detection(self, road_index: int, detection: dict):
        # increment confirm counts for this road, reset others
        for i in range(1,5):
            if i == road_index:
                self._confirm_counts[i] += 1
                self._clear_counts[i] = 0
            else:
                self._confirm_counts[i] = 0
                self._clear_counts[i] += 1

        if self._confirm_counts[road_index] >= self.confirm_frames:
            # activate emergency
            self.activate_emergency(f"Road {road_index}", detection.get("vehicle_type"), detection.get("confidence"))

    def process_no_detection(self, road_index: int):
        # increment clear counts; if current emergency present and cleared for all required frames, clear it
        if self.is_emergency and self.current_emergency:
            # which road was emergency
            for i in range(1,5):
                if f"Road {i}" == self.current_emergency.get("road"):
                    self._clear_counts[i] += 1
                    if self._clear_counts[i] >= self.clear_frames:
                        self.clear_emergency()

    def activate_emergency(self, road: str, vehicle: str, confidence: float, simulated: bool=False):
        self.is_emergency = True
        self.current_emergency = {"road": road, "vehicle": vehicle, "confidence": confidence, "simulated": simulated, "since": time.time()}

    def clear_emergency(self):
        self.is_emergency = False
        self.current_emergency = None
        # reset counters
        self._confirm_counts = {1:0,2:0,3:0,4:0}
        self._clear_counts = {1:0,2:0,3:0,4:0}
