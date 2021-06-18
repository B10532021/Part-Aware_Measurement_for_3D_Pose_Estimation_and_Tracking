import numpy as np
import cv2

class KalmanFilter(object):
    def __init__(self, pt3d, Hz=25):
        self.Hz = Hz # Frequency of Vision System
        dt = 1.0/Hz
        v = dt
        a = 0.5 * (dt**2)
        sp = 0.1

        self.kalman = cv2.KalmanFilter(9,3,0)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, v, 0, 0, a, 0, 0],
            [0, 1, 0, 0, v, 0, 0, a, 0],
            [0, 0, 1, 0, 0, v, 0, 0, a]
            ],np.float32)

        self.kalman.transitionMatrix = np.array([
                [1, 0, 0, v, 0, 0, a, 0, 0],
                [0, 1, 0, 0, v, 0, 0, a, 0],
                [0, 0, 1, 0, 0, v, 0, 0, a],
                [0, 0, 0, 1, 0, 0, v, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, v, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, v],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ],np.float32)

        self.kalman.processNoiseCov = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ],np.float32) * 0.007

        self.kalman.measurementNoiseCov = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0 ,1]
            ],np.float32) * sp

        self.kalman.statePre = np.array([
                [np.float32(pt3d[0])], [np.float32(pt3d[1])], [np.float32(pt3d[2])]
                , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
                , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
            ])
    
    def predict(self, pt3d=None):
        if pt3d is not None:
            mp = np.array([
                [np.float32(pt3d[0])],
                [np.float32(pt3d[1])],
                [np.float32(pt3d[2])]
            ])
            self.kalman.correct(mp)

        tp = self.kalman.predict()
        return tp[:3].flatten()