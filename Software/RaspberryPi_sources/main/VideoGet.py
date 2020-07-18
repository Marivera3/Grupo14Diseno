# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoGet.py
import threading
import cv2 as cv
import imutils
from imutils.video import FPS

class VideoGet(threading.Thread):

    def __init__(self, src=0, name='Video Getter'):
        super().__init__(name=name)
        self.stream = cv.VideoCapture()
        self.stream.open(src, apiPreference=cv.CAP_V4L2)
        (self.rval, self.frame) = self.stream.read()
        self.fps = FPS().start()
        self.stopped = False


    def run(self):
        self.stopped = False
        print('[INFO] Thread started: {}...'.format(threading.current_thread().name))
        while not self.stopped:
            if self.rval:
                (self.rval, self.frame) = self.stream.read()
                self.fps.update()
            else:
                self.stop()

        print(f'[INFO] Releasing video capture...')
        self.stream.release()

    def stop(self):
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        self.stopped = True
        self.rval = False
