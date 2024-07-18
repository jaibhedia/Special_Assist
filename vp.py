import cv2
import mediapipe as mp
import numpy as np
import os
import math
from gtts import gTTS
import pygame
import pytesseract
import io

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def play_audio(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
        continue

# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

# Image that will contain the drawing and then passed to the camera image
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Getting all header images in a list
folderPath = './Header'
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Load ASL images and resize them, with error handling
asl_images = {}
image_files = {
    'water': 'asl_water.png',
    'drink water': 'paytm.png',
    'have dinner': 'asl_have_dinner.png',
    'lunch': 'asl_lunch.png',
    'emergency': 'asl_emergency.png',
    'hi': 'asl_hi.png'
}

for key, file in image_files.items():
    if os.path.exists(file):
        asl_images[key] = cv2.resize(cv2.imread(file), (300, 300))
    else:
        print(f"Warning: {file} not found")

# Presettings:
header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20 # Thickness of the painting
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Getting all hand points coordinates
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                # Only go through the code when a hand is detected
                if len(points) != 0:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12] # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20] # Pinky

                    ## Checking which fingers are up
                    fingers = []
                    # Checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # The rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    ## Selection Mode - Two fingers are up (thumb down, ring and pinky down)
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in [0, 3, 4]):
                        xp, yp = [x1, y1]

                        # Selecting the colors and the eraser on the screen
                        if(y1 < 125):
                            if(170 < x1 < 295):
                                header = overlayList[0]
                                drawColor = (0, 0, 255)
                            elif(436 < x1 < 561):
                                header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif(700 < x1 < 825):
                                header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif(980 < x1 < 1105):
                                header = overlayList[3]
                                drawColor = (0, 0, 0)

                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    ## Stand by Mode - Checking when the index and the pinky fingers are open and dont draw
                    nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        # The line between the index and the pinky indicates the Stand by Mode
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                        xp, yp = [x1, y1]

                    ## Draw Mode - One finger is up
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        # The circle in the index finger indicates the Draw Mode
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        # Draw a line between the current position and the last position of the index finger
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        # Update the last position
                        xp, yp = [x1, y1]

                    ## Clear the canvas when the hand is closed
                    if all(fingers[i] == 0 for i in range(0, 5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]

                    ## Adjust the thickness of the line using the index finger and thumb
                    selecting = [1, 1, 0, 0, 0] # Selecting the thickness of the line
                    setting = [1, 1, 0, 0, 1]   # Setting the thickness chosen
                    if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):

                        # Getting the radius of the circle that will represent the thickness of the draw
                        # using the distance between the index finger and the thumb.
                        r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                        
                        # Getting the middle point between these two fingers
                        x0, y0 = [(x1+x3)/2, (y1+y3)/2]
                        
                        # Getting the vector that is orthogonal to the line formed between
                        # these two fingers
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        # Normalizing it 
                        mod_v = math.sqrt(v1**2 + v2**2)
                        v1, v2 = [v1/mod_v, v2/mod_v]
                        
                        # Draw the circle that represents the draw thickness in (x0, y0) and orthogonaly 
                        # translated c units
                        c = 3 + r
                        x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                        cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                        # Setting the thickness chosen when the pinky finger is up
                        if fingers[4]:                        
                            thickness = r
                            cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)

                        xp, yp = [x1, y1]

                    ## Check for specific signs and display corresponding ASL image and play audio
                    if fingers == [0, 1, 0, 0, 1]: # Index and pinky fingers up, rest down
                        play_audio("Ondu Glass Neeru Kodi")
                        image[100:400, 100:400] = asl_images.get('water', np.zeros((300, 300, 3), np.uint8))
                        xp, yp = [x1, y1]
                    elif fingers == [1, 0, 0, 0, 1]: # Thumb and pinky fingers up, rest down
                        play_audio("Paytm mein saw rupaye prapt hue hain")
                        image[100:400, 100:400] = asl_images.get('drink water', np.zeros((300, 300, 3), np.uint8))
                        xp, yp = [x1, y1]
                    elif fingers == [1, 1, 0, 0, 0]: # Thumb and index fingers up, rest down
                        play_audio("Khana Chahiye")
                        image[100:400, 100:400] = asl_images.get('lunch', np.zeros((300, 300, 3), np.uint8))
                        xp, yp = [x1, y1]
                    elif fingers == [0, 0, 0, 1, 0]: # Ring finger up, rest down
                        play_audio("Emergency")
                        image[100:400, 100:400] = asl_images.get('emergency', np.zeros((300, 300, 3), np.uint8))
                        xp, yp = [x1, y1]
                    elif fingers == [1, 0, 0, 0, 0]: # Thumb up, rest down
                        play_audio("Namaskara")
                        image[100:400, 100:400] = asl_images.get('hi', np.zeros((300, 300, 3), np.uint8))
                        xp, yp = [x1, y1]

        # Setting the header in the video
        image[0:125, 0:width] = header

        # The image processing to produce the image of the camera with the draw made in imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Perform text recognition
        try:
            text = pytesseract.image_to_string(imgCanvas, config='--psm 6').strip().lower()
            if text in asl_images:
                play_audio(text.capitalize())
                image[100:400, 100:400] = asl_images[text]
        except pytesseract.TesseractNotFoundError:
            print("Tesseract OCR is not installed or not found in your PATH.")

        cv2.imshow('MediaPipe Hands', img)

        # Check for keyboard input
        if cv2.waitKey(3) & 0xFF == ord('w'):
            play_audio("w")

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
