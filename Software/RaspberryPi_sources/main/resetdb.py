import mongoengine as me
import time
from User.User import PersonRasp
c = 0
erase = True
while erase:

    try:
        me.connect('Rasp', host='127.0.0.1', port=27017)
        for doc in PersonRasp.objects():
            if doc.surname == "campeny":
                doc.delete()
        erase = False
    except Exception as e:
        print(e)
        print("Unknown cleaning fails...")
        c += 1
        time.sleep(0.01)
        if c > 50:
            print("Couldt connect to mongo")
            erase = False
