import cv2
import mediapipe as mp
import numpy as np
import time
from pythonosc import udp_client

# ---- OSC Setup ----
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

# ---- Mediapipe Setup ----
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def get_landmark_coords(results):
    """Extract 3D landmark coordinates as numpy array."""
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])


def compute_focus_score(baseline, current):
    """Compute focus score (0-1) based on deviation from baseline."""
    if current is None:
        return 0.0
    deviation = np.linalg.norm(current - baseline, axis=1)

    # Give iris landmarks higher weighting.
    for landmark_number in range(469, 478):
        deviation[landmark_number] *= 375

    avg_dev = np.mean(deviation)
    score = max(0.0, 1.0 - avg_dev * 50)
    return score


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        # ---- Initial Calibration ----
        print("Calibrating for 5 seconds... please stay relaxed.")
        all_points = []
        start = time.time()
        while time.time() - start < 5:
            success, frame = cap.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            points = get_landmark_coords(results)
            if points is not None:
                all_points.append(points)
            cv2.imshow("Calibration", cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        if not all_points:
            raise RuntimeError("Calibration failed: no landmarks found.")

        baseline = np.mean(np.stack(all_points), axis=0)
        print("Calibration complete. Tracking focus...")

        # ---- Tracking Variables ----
        last_update = time.time()
        alpha = 0.5  # recalibration rate
        update_interval = 1.0  # seconds

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            current = get_landmark_coords(results)

            # Every few seconds: compute score and recalibrate
            if time.time() - last_update > update_interval and current is not None:
                score = compute_focus_score(baseline, current)
                osc_client.send_message("/focus", f"{score:.3f}")
                print(f"Focus: {score:.3f}")

                # Gradually update baseline (auto recalibration)
                baseline = (1 - alpha) * baseline + alpha * current

                last_update = time.time()

            # Draw face mesh
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

            cv2.imshow("Focus Tracker", cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()