from app import app, socketio
import logging
import sys

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print ('connected to server at localhost')
    socketio.run(app)
    