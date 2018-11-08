from flask import Flask
from flask_socketio import SocketIO

app= Flask(__name__)
app.config['DEBUG'] = True

socketio= SocketIO(app)


from app.views import home