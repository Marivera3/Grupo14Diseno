import threading
import requests
import json
import time
import datetime
import mongoengine as me
from User.User import PersonRasp


class CheckDB2(threading.Thread):

        def __init__(self, name, seconds=2):
            super().__init__(name=name)
            self.__url_DB     = "http://server4diseno.duckdns.org:1227/DB"
            self.__url_Number = "http://server4diseno.duckdns.org:1227/Number"
            self.stopped    = False
            self.actual_dbsize = 0
            self.prev_time = datetime.datetime.utcnow()
            self.has_changes = False
            self.data = None
            self.seconds = seconds
            ## for mongo
            self.host = '192.168.0.25'
            # self.host = '192.168.0.242'
            self.port = 27017
            # self.db = 'test1'
            self.db = 'Rasp'
            self.newnames = []


        def run(self):
            cpt = 0
            while True:
                if (datetime.datetime.utcnow() - self.prev_time).total_seconds() > self.seconds:
                    self.prev_time = datetime.datetime.utcnow()
                    try:
                        myfile = requests.get(self.__url_Number)
                        if myfile.status_code == 200:
                            new_size = int(myfile.content.decode())
                            if new_size > self.actual_dbsize:
                                database = requests.get(self.__url_DB)
                                # print(f'DB: {database.content.decode()}')
                                if database.status_code == 200:
                                    self.actual_dbsize = int(myfile.content.decode())
                                    self.has_changes = True
                                    self.data = json.loads(database.content.decode())
                                    self.tomongo()
                                    # print(self.data[0]['idlist'][0])
                            else:
                                self.has_changes = False
                        cpt = 0
                    except:
                        print('[INFO] Could no connect to Server Database ')
                        cpt += 1
                        time.sleep(cpt/10.0 + 1)
                        if cpt > 50:
                            cpt = 0
                    time.sleep(1)


        def tomongo(self):
            cpt = 0
            connected = False
            while not connected:

                try:

                    print('[INFO] Searching for unkown in Mongo...')
                    db_client = me.connect(self.db, host=self.host , port=self.port)
                    for doc in PersonRasp.objects():
                        for docweb in self.data:
                            if doc.Idrasp in docweb['idlist']:
                                self.newnames.append([doc.name, (docweb.name, docweb.surname)])

                    connected = True
                    # print(f'New names: {self.newnames}')
                    print('[INFO] Connection succed')
                    db_client.close()
                    cpt = 0

                except Exception as e:
                    print(e)
                    # Falta fijar la excepcion
                    print('[INFO] CheckDB - Connection fails')
                    connected = False
                    cpt += 1
                    time.sleep(cpt/10.0 + 1)
                    if cpt > 50:
                        cpt = 1



# CheckDB2("check").start()
