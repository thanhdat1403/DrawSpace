import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Đặt độ rộng khung hình
cap.set(4, 600)  # Đặt chiều cao khung hình

# Khởi tạo Mediapipe
mpHands = mp.solutions.hands #mpHands.Hands: Khởi tạo đối tượng phát hiện tay
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)#min_detection_confidence=0.7: Ngưỡng để phát hiện tay (0.7 tức là độ tin cậy phải >= 70%).
mpDraw = mp.solutions.drawing_utils # mpDraw: Công cụ để vẽ các điểm và đường kết nối trên tay

# Màu sắc và nhãn
colors = [
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (226, 43, 138), # Purple
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (0, 0, 0),      # Erase
]
labels = ["Red", "Blue", "Green", "Purple", "Yellow", "Cyan", "Erase"]
color = colors[0]

# Các biến vẽ
xp, yp = 0, 0
brush_thickness = 10
erase_thickness = 50
header_height = 100

# Khởi tạo canvas
canvas = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Lật khung hình
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    # Vẽ thanh chọn màu
    section_width = w // len(colors)
    for i, (label, col) in enumerate(zip(labels, colors)):
        cv2.rectangle(frame, (i * section_width, 0), ((i + 1) * section_width, header_height), col, -1)
        text_color = (255, 255, 255) if col != (255, 255, 255) else (0, 0, 0)
        cv2.putText(frame, label, (i * section_width + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Xử lý Mediapipe
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Toạ độ ngón trỏ và ngón giữa
            x1, y1 = landmarks[8][1:]  # Ngón trỏ
            x2, y2 = landmarks[12][1:]  # Ngón giữa

            # Chế độ chọn màu
            if y1 < header_height:
                section_index = min(max(x1 // section_width, 0), len(colors) - 1)
                color = colors[section_index]

            # Chế độ vẽ
            elif y1 < landmarks[6][2] and y2 > landmarks[10][2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                thickness = erase_thickness if color == (0, 0, 0) else brush_thickness
                cv2.line(canvas, (xp, yp), (x1, y1), color, thickness)
                xp, yp = x1, y1

            else:
                xp, yp = 0, 0

    # Kết hợp canvas với frame
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    # Hiển thị khung hình
    cv2.imshow("Virtual Painter", frame)
    cv2.imshow("Canvas", canvas)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
