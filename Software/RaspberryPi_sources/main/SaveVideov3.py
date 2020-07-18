import threading
import cv2
import datetime
from subprocess import Popen, PIPE
import os


class SaveVideo(threading.Thread):

    def __init__(self, name, vg, pathIn, pathOut, fps=25, encode_quality=95):
        # '-t', str(time)
        super().__init__(name=name)
        print('[INFO] Conexion for Video Creation using ffmpeg...')
        self.vg             = vg
        self.pathOut        = pathOut
        self.pathIn         = pathIn
        self.encode_param   = [cv2.IMWRITE_JPEG_QUALITY, encode_quality]
        self.stopped        = False
        self.folder         = 1
        self.fps            = fps



    def setprocess(self):
        pathOut = self.pathIn + str(self.folder) + self.pathOut
        print(pathOut)
        process = Popen(['ffmpeg',
                                        '-y', # overwrite output file
                                        '-async', '0',
                                        '-f', 'image2pipe',
                                        '-vcodec', 'mjpeg',
                                        '-use_wallclock_as_timestamps','1',
                                        '-i', '-', # The input comes from a pipe
                                        '-an', # Tells ffmpeg not to expect any audio
                                        '-loglevel', 'error',
                                        '-pix_fmt', 'yuv420p',
                                        '-vcodec', 'mpeg4',
                                        '-q','5', # Variable Bit Rate: number from 1-31, with 1 being highest quality/largest filesize and 31 being the lowest quality/smallest filesize
                                        '-r', str(self.fps),
                                        '-t', '00:00:20',
                                        # '-t', '00:59:59',
                                        pathOut,
                                        '-async', '1',
                                        '-vsync', '1'
                                        ], stdin=PIPE,
                                        )
        return process


    def run(self):
        process = self.setprocess()
        while not self.stopped and self.vg.rval:
            data = cv2.imencode('.jpg', self.vg.frame, self.encode_param)[1].tobytes()
            if process.poll() == None:
                # print('still alive')
                try:
                    process.stdin.write(data)
                except BrokenPipeError:
                    pass
            else:
                print('terminated')
                self.folder += 1

                if self.folder > 24:
                    self.folder = 1
                    # self.stop()
                # else:
                #     process = self.setprocess()
                # rutina para eliminar videos pasados
                process = self.setprocess()
                num = self.folder - 7
                if num < 0:
                    num = num + 24
                myfile=self.pathIn + str(num) + self.pathOut
                if os.path.isfile(myfile):
                    os.remove(myfile)

            #


        print(f'[INFO] Saving Video to {self.pathOut}...')
        # process.stdin.close()

        # self.process.stdin.close()
        # print(f'[INFO] Saving Video to {self.pathOut}...')


    def stop(self):
        self.stopped = True
        # print(f'[INFO] Saving Video to {self.pathOut}...')
        # self.process.stdin.close()
