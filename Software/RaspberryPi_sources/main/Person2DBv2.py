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
# from pymongo import MongoClient
from User.User import PersonRasp

class Person2DB(threading.Thread):

        def __init__(self, paquete):
            # '-t', str(time)
            super().__init__()

            # self.host = 'mongodb://grupo14.duckdns.org:1226/Rasp'
            # self.host = '192.168.0.25'
            self.host = '127.0.0.1'
            # self.host = '192.168.0.242'
            self.port = 27017
            # self.db = 'test1'
            self.db = 'Rasp'
            self.stopped = False
            self.likelihood, self.pic, self.rec, self.out, self.inn, self.name = paquete
            # self.likelihood, self.pic, self.rec, self.out, self.name = paquete
            self.timenow = dt.datetime.now()


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

        def modifypersondb(name, surname, idh, last_out, likelihood):
            # rutine with pymongo
            pass


        def run(self):
            cpt = 0
            connected = False
            if self.rec:
                jpeg_frame = ""
            else:
                # jpeg_frame = pickle.dumps(cv2.imencode('.jpg', self.pic, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tostring())
                jpeg_frame = base64.b64encode(cv2.imencode('.png', self.pic, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()).decode('ascii')
                # print(jpeg_frame)
            while not connected:

                try:

                    print('[INFO] Connecting to DB mongoengine...')
                    db_client = me.connect(self.db, host=self.host , port=self.port)
                    # db_client = me.connect('test1', host='192.168.0.242', port=27017)
                    # time.sleep(0.05)
                    connected = True
                    print("Nombre en DB")
                    print(self.name)

                    # pickled = codecs.encode(pickle.dumps(item[0]), "base64").decode()
                    if self.name.split(' ')[0] == 'unknown':
                        name = self.name
                        surname = ''
                    else:
                        aux = self.name.split('_')
                        name = aux[0]
                        surname = aux[1]

                    if not self.out:
                        self.addperson2db(name=name, surname=surname, is_recongized=self.rec,
                                    last_in=self.timenow,
                                    picture=jpeg_frame, likelihood=self.likelihood)
                    else:
                        # rutina de salida
                        if PersonRasp.objects():
                            for doc in PersonRasp.objects():
                                if doc['name'] == name and doc['surname'] == surname:
                                    doc.update(set__last_out=self.timenow,
                                                set__is_recongized=self.rec,
                                                set__likelihood=self.likelihood)



                    # time.sleep(0.05)
                    print('[INFO] Connection succed')
                    db_client.close()
                    cpt = 0

                except Exception as e:
                    print(e)
                    # Falta fijar la excepcion
                    print('[INFO] Connection fails')
                    connected = False
                    cpt += 1
                    time.sleep(cpt/10.0 + 1)
                    if cpt > 50:
                        cpt = 1
