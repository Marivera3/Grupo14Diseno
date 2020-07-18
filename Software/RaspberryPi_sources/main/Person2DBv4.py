import threading
import codecs
import hashlib
import io
import pickle
import queue
import codecs
import time
import cv2
import base64
import datetime as dt
import mongoengine as me
from collections import deque
import requests
# from pymongo import MongoClient
from User.User import PersonRasp

class Person2DB(threading.Thread):

        def __init__(self):
            # '-t', str(time)
            super().__init__()

            # self.host = 'mongodb://grupo14.duckdns.org:1226/Rasp'
            # self.host = '192.168.0.25'
            self.host = '127.0.0.1'
            # self.host = '192.168.0.242'

            self.port = 27017
            # self.db = 'test1'
            self.db = 'Rasp'
            self.queue = deque()
            self.stopped = False
            # self.likelihood, self.pic, self.rec, self.out, self.inn, self.name = paquete
            # self.likelihood, self.pic, self.rec, self.out, self.name = paquete
            # self.timenow = dt.datetime.now()


        def addperson2db(self, name, surname, is_recongized, last_in, picture, likelihood):
            idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
            # only entries, via mongoengine
            if is_recongized:
                PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                            is_recognized=is_recongized, likelihood=likelihood).save()

            else:
                PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                            is_recognized=is_recongized, likelihood=likelihood,
                            picture=picture).save()

        def addperson2dbregmode(self, name, surname):
            idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
            # only entries, via mongoengine
            PersonRasp(idrasp= idh,name=name, surname=surname,
            is_recognized=True, is_trained=True).save()



        def modifypersondb(name, surname, idh, last_out, likelihood):
            # rutine with pymongo
            pass

        def stop(self):
            self.stopped = True

        def run(self):

            waiting2connect = False

            while not self.stopped or len(self.queue) > 0:
                c = 0
                while not waiting2connect and len(self.queue) > 0:
                    elem = ""
                    try:
                        db_client = me.connect(self.db, host=self.host , port=self.port)
                        waiting2connect = True
                        elem = self.queue.popleft()
                        if len(elem) > 2:
                            likelihood, pic, rec, out, inn, rawname, timenow = elem
                            if rec:
                                jpeg_frame = ""
                            else:
                                # jpeg_frame = pickle.dumps(cv2.imencode('.jpg', self.pic, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tostring())
                                jpeg_frame = base64.b64encode(cv2.imencode('.png', pic, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()).decode('ascii')
                                # print(jpeg_frame)

                            if rawname.split(' ')[0] == 'unknown':
                                name = rawname
                                surname = ''
                            else:
                                aux = rawname.split('_')
                                name = aux[0]
                                surname = aux[1]

                            if not out:
                                self.addperson2db(name=name, surname=surname, is_recongized=rec,
                                            last_in=timenow,
                                            picture=jpeg_frame, likelihood=likelihood)
                            else:
                                # rutina de salida
                                if PersonRasp.objects():
                                    for doc in PersonRasp.objects():
                                        if doc['name'] == name and doc['surname'] == surname:
                                            doc.update(set__last_out=timenow,
                                                        set__is_recongized=rec,
                                                        set__likelihood=likelihood)
                        else:
                            # reg mode
                            name, surname = elem
                            self.addperson2dbregmode(name=name, surname=surname)
                        # time.sleep(0.05)
                        print('[INFO] Connection succed')

                    except Exception as e:
                        print(e)
                        # Falta fijar la excepcion
                        print('[INFO] Connection fails')
                        c += 1
                        time.sleep(0.01)
                        waiting2connect = False
                        if elem:
                            self.queue.appendleft(elem)
                        if c > 15:
                            waiting2connect = True


                try:
                    response = requests.get('https://www.google.com/')
                    waiting2connect = False
                except:
                    print('no internet')
                    time.sleep(5)
