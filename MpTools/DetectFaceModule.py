import cv2
import mediapipe as mp


class FaceDetect:
    def __init__(self, min_detection_confidence=0.5):

        self.min_detection_confidence = min_detection_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.min_detection_confidence,)

    def detect_face(self, mat, draw=True):
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        results = self.face.process(rgb)
        if draw:
            if results.detections:
                for handlm in results.detections:
                    self.mpDraw.draw_detection(mat, handlm)
        return mat
