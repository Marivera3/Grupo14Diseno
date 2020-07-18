import mongoengine as me
import datetime


class FramesVideo(me.Document):


    created_at          = me.DateTimeField(default=datetime.datetime.utcnow)
    frame               = me.BinaryField(required=True)
    equivalent2         = me.IntField(min_value=1, default=1)
    maincamera          = me.BooleanField(default=True)
