import threading
import codecs
import hashlib
import io
import pickle
import queue
import codecs
import time
import cv2
import datetime as dt
import mongoengine as me
# from pymongo import MongoClient
from User.User import PersonRasp

class Person2DBv3(threading.Thread):

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
            # self.stopped = False
            self.name, self.surname = paquete
            # self.likelihood, self.pic, self.rec, self.out, self.name = paquete
            # self.timenow = dt.datetime.utcnow()


        def addperson2db(self, name, surname):
            idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
            # only entries, via mongoengine
            PersonRasp(idrasp= idh,name=name, surname=surname,
            is_recognized=True, is_trained=True).save()



        def run(self):
            cpt = 0
            connected = False
            while not connected:

                try:

                    print('[INFO] Connecting to DB mongoengine...')
                    db_client = me.connect(self.db, host=self.host , port=self.port)
                    # db_client = me.connect('test1', host='192.168.0.242', port=27017)
                    # time.sleep(0.05)
                    connected = True
                    self.addperson2db(name=self.name, surname=self.surname)




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
