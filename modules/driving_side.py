import cv2

def analyze_driving_side(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_left = total_right = 0

    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        left = edges[:, :width // 2].sum()
        right = edges[:, width // 2:].sum()
        total_left += left
        total_right += right

    cap.release()
    return "left" if total_left > total_right else "right"
