import threading
import socket
import requests
import json

class RegMode(threading.Thread):

    def __init__(self, host, port):
        super().__init__()
        self.Register_mode = False
        self.Register_id = 0
        server_address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(server_address)
        self.reset()


    def reset(self):
        self.Register_mode = False
        self.Register_id = 0

    def run(self):
        self.sock.listen()
        while True:
            socket_cliente, _ = self.sock.accept()
            self.Register_mode = True
            myfile = requests.get('http://server4diseno.duckdns.org:1227/Waitingroom')
            if myfile.status_code == 200:
                new_size = json.loads(myfile.content.decode())
            self.Register_id = new_size[0]['name']
            self.name = new_size[0]['name']
            self.surname = new_size[0]['surname']
            socket_cliente.close()
