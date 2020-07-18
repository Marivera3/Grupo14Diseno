import threading
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from imutils.video import FPS
from queue import Queue

class FrameProcessing(threading.Thread):

    def __init__(self, name, queue_video, queue_frame, detector, embedder, recognizer, le, fa):
        super().__init__(name=name)
        self.queue_video    = queue_video
        self.stopped        = False
        self.fps            = FPS().start()
        self.queue_frames   = queue_frame
        self.detector       = detector
        self.embedder       = embedder
        self.recognizer     = recognizer
        self.le             = le

    def run(self):
        self.stopped = False
        print('[INFO] Thread started: {}...'.format(threading.current_thread().name))
        while not self.stopped:
            if self.queue_video.qsize() > 0:
                frame = self.queue_video.get()
                face_data = acquire_frame(self.detector, self.embedder, frame , self.recognizer, self.le, 0.5, 0.65, fa)
                self.queue_frames.put(face_data)
                self.fps.update()


    def stop(self):
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        self.stopped = True
