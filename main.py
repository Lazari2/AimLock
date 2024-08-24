import cv2
import numpy as np
import pyautogui
from pynput import keyboard
from ultralytics import YOLO

model = YOLO('C:/opencv/aimlock/runs/detect/train/weights/best.pt')

screen_width, screen_height = pyautogui.size()
center_screen_x = screen_width // 2
center_screen_y = screen_height // 2

point_color = (0, 0, 255)  
ball_color = (255, 0, 0)   

room_width = 2000
room_height = 1800

ball_pos = [center_screen_x + 200, center_screen_y + 100]  

ball_radius = 20
room_x_limit = room_width - ball_radius
room_y_limit = room_height - ball_radius

def draw_point_on_screen(screen):
    cv2.circle(screen, (center_screen_x, center_screen_y), 5, point_color, -1)

def draw_blue_ball(screen, ball_pos):
    int_ball_pos = (int(ball_pos[0]), int(ball_pos[1])) 
    cv2.circle(screen, int_ball_pos, ball_radius, ball_color, -1)

mouse_sensitivity = 0.5

def move_screen_based_on_mouse(mouse_pos, ball_pos, room_offset):
    dx = (mouse_pos[0] - center_screen_x) * mouse_sensitivity
    dy = (mouse_pos[1] - center_screen_y) * mouse_sensitivity

    room_offset[0] += dx
    room_offset[1] += dy

    room_offset[0] = max(0, min(room_offset[0], room_width - screen_width))
    room_offset[1] = max(0, min(room_offset[1], room_height - screen_height))

    ball_pos[0] = max(ball_radius, min(ball_pos[0] - dx, room_x_limit))
    ball_pos[1] = max(ball_radius, min(ball_pos[1] - dy, room_y_limit))

    return room_offset

# Função Aimbot
def center_ball(ball_pos, room_offset):
    dx = center_screen_x - ball_pos[0]
    dy = center_screen_y - ball_pos[1]

    ball_pos[0] = center_screen_x
    ball_pos[1] = center_screen_y

    room_offset[0] -= dx
    room_offset[1] -= dy

    return room_offset

#Botão Aimbot
def on_press(key):
    try:
        if key == keyboard.Key.shift:
            print("Shift")
            room_offset[:] = center_ball(ball_pos, room_offset)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()


room = np.ones((room_height, room_width, 3), dtype=np.uint8) * 255

# Desenho de grade do fundo pra ajudar visualização
def draw_grid(screen, grid_size=50):
    for x in range(0, screen.shape[1], grid_size):
        cv2.line(screen, (x, 0), (x, screen.shape[0]), (200, 200, 200), 1)
    for y in range(0, screen.shape[0], grid_size):
        cv2.line(screen, (0, y), (screen.shape[1], y), (200, 200, 200), 1)

draw_grid(room)

room_offset = [0, 0]

while True:
    mouse_x, mouse_y = pyautogui.position()

    room_offset = move_screen_based_on_mouse((mouse_x, mouse_y), ball_pos, room_offset)

    int_room_offset_x = int(room_offset[0])
    int_room_offset_y = int(room_offset[1])

    visible_room = room[int_room_offset_y:int_room_offset_y + screen_height, int_room_offset_x:int_room_offset_x + screen_width].copy()

    draw_point_on_screen(visible_room)
    draw_blue_ball(visible_room, ball_pos)

    results = model(visible_room)

    for result in results:
        for box in result.boxes:
            if box.conf > 0.7:  
                print(f"Objeto: {box.xyxy}")

                # caixinha
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(visible_room, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Aim Lock Simulation', visible_room)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
listener.stop()
