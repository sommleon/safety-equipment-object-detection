from ultralytics import YOLO
from datetime import datetime
import cv2 as cv
import time
import math
import random

fontProperties = {
    'fontFace':  cv.FONT_HERSHEY_SIMPLEX,
    'fontScale': 0.5,
    'fontColor': (255, 255, 255),
    'fontThickness': 2,
    'fontLineType': cv.LINE_AA
}

classes = [
    'Helm',
    'Warnweste',
    'Kopf',
    'Person'
]

boundingBoxColors = {
    'Helm': (0, 255, 0),
    'Warnweste': (255, 255, 255),
    'Kopf': (0, 0, 255),
    'Person': (255, 0, 0)
}

frameTimes = {
    'new': 0,
    'previous': 0
}

classCounters = {
    'Helm': 0,
    'Warnweste': 0,
    'Kopf': 0,
    'Person': 0
}

# 92 Width X 143 Height
warningSign = cv.imread('warning_sign.png')


def openCapture():
    # Capture standard webcam
    # capture = cv.VideoCapture(0)
    # Optional for emulating video stream
    capture = cv.VideoCapture('video_sample.mp4')
    # Resolution Width ID 3
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    # Resolution Height ID 4
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    return capture


def closeCapture(capture):
    capture.release()
    cv.destroyAllWindows()


def saveSafetyViolation(frame, x1, x2, y1, y2):
    timeTag = datetime.now().strftime('%Y%m%d-%H%M%S')
    screenshot = frame[y1:y2, x1:x2]
    screenshot = cv.resize(screenshot, (230, 320))
    cv.imwrite(f'./safety_violations/img{timeTag}-{random.randint(1,100)}.png', screenshot)


def drawBoundingBox(frame, className, confidence, x1, y1, x2, y2):
    boundingBoxColor = boundingBoxColors[className]
    cv.rectangle(frame, (x1, y1), (x2, y2), boundingBoxColor, fontProperties['fontThickness'],
                  fontProperties['fontLineType'])
    cv.putText(frame, f'{className} {confidence:.2f}%', (x1, y1 - 10), fontProperties['fontFace'],
                fontProperties['fontScale'], fontProperties['fontColor'], fontProperties['fontThickness'],
                fontProperties['fontLineType'])


def displayHUD(frame, capture, fps):
    cv.putText(frame, f'{fps} FPS', (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)) - 60, 20), fontProperties['fontFace'],
               fontProperties['fontScale'], fontProperties['fontColor'], fontProperties['fontThickness'],
               fontProperties['fontLineType'])
    cv.putText(frame, 'REC', (12, 20), fontProperties['fontFace'], fontProperties['fontScale'],
               fontProperties['fontColor'], fontProperties['fontThickness'], fontProperties['fontLineType'])
    cv.circle(frame, (55, 15), 6, (0, 0, 255), -1, fontProperties['fontLineType'])
    cv.putText(frame, f'Schutzhelm: {classCounters["Helm"]}', (10, int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) - 80),
               fontProperties['fontFace'], fontProperties['fontScale'], fontProperties['fontColor'],
               fontProperties['fontThickness'], fontProperties['fontLineType'])
    cv.putText(frame, f'Warnweste: {classCounters["Warnweste"]}', (10, int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) - 60),
               fontProperties['fontFace'], fontProperties['fontScale'], fontProperties['fontColor'],
               fontProperties['fontThickness'], fontProperties['fontLineType'])
    cv.putText(frame, f'Schutzlos: {classCounters["Kopf"]}', (10, int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) - 40),
               fontProperties['fontFace'], fontProperties['fontScale'], fontProperties['fontColor'],
               fontProperties['fontThickness'], fontProperties['fontLineType'])

    divisor = int(classCounters["Helm"]) + int(classCounters["Kopf"])
    if divisor == 0:
        cv.putText(frame, 'Schutzpflicht: N/A', (10, int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) - 20),
                   fontProperties['fontFace'], fontProperties['fontScale'], fontProperties['fontColor'],
                   fontProperties['fontThickness'], fontProperties['fontLineType'])
    else:
        quote = math.ceil(int(classCounters["Helm"]) / divisor * 100)
        cv.putText(frame, f'Schutzpflicht: {quote}%', (10, int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) - 20),
                   fontProperties['fontFace'], fontProperties['fontScale'], fontProperties['fontColor'],
                   fontProperties['fontThickness'], fontProperties['fontLineType'])

    if classCounters["Kopf"] >= 1:
        # 92 Width X 143 Height
        x1 = int(capture.get(cv.CAP_PROP_FRAME_WIDTH) - 102)
        x2 = int(capture.get(cv.CAP_PROP_FRAME_WIDTH) - 10)
        y1 = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT) - 153)
        y2 = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT) - 10)
        frame[y1:y2, x1:x2] = warningSign


def main():
    capture = openCapture()
    timer = 0
    model = YOLO('./yolov8n_helmet_vest.pt')
    while capture.isOpened():
        frameTimes['new'] = time.time()
        timeElapsed = (frameTimes['new'] - frameTimes['previous'])
        frameTimes['previous'] = frameTimes['new']
        fps = math.ceil(1 / timeElapsed)
        for key in classCounters:
            classCounters[key] = 0
        success, frame = capture.read()
        if not success:
            break
        results = model.predict(frame, stream=True)
        for result in results:
            boundingBoxes = result.boxes
            for boundingBox in boundingBoxes:
                confidence = boundingBox.conf[0] * 100
                if confidence >= 40:
                    classNumber = int(boundingBox.cls[0])
                    className = classes[classNumber]
                    x1, y1, x2, y2 = boundingBox.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if className == "Kopf":
                        timer += timeElapsed
                        if timer >= 10:
                            saveSafetyViolation(frame, x1, x2, y1, y2)
                    classCounters[className] += 1
                    drawBoundingBox(frame, className, confidence, x1, y1, x2, y2)
        if timer >= 10:
            timer = 0
        displayHUD(frame, capture, fps)
        cv.imshow('Safety Detection Software', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    closeCapture(capture)
    return 0


if __name__ == "__main__":
    main()