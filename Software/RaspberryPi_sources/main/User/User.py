import mongoengine as me
import datetime


class PersonRasp(me.Document):


    idrasp              = me.StringField(unique=True, required=True)
    created_at          = me.DateTimeField(default=datetime.datetime.utcnow)
    name                = me.StringField(max_length=255, default='')
    surname             = me.StringField(max_length=255, default='')
    last_in             = me.DateTimeField()
    last_out            = me.DateTimeField()
    is_recognized       = me.BooleanField(default=False)
    seralize_pic        = me.StringField() #
    picture             = me.StringField() # RGB picture in bytes(?)
    likelihood          = me.DecimalField()
    is_trained          = me.BooleanField(default=False)
