from flask import render_template
from flask_socketio import emit
from app import app, socketio
import json

@app.route('/')

def home():
    return render_template('home.html')

users=[]

@socketio.on('connect')
def connect():
    print('User connected') 

@socketio.on('addUser')
def addUser(user):
    users.append(user)
    emit('users',users, broadcast=True)

@socketio.on('ganify')
def RunGan(data):
    print(data) #here we call Renauds program

    #newData='{d1:"many voxels",d2:"many voxels",d3:"a gazillion voxels"}'
    newData= 'this is python talking to JS - Hi JS!' 
    emit('executedGAN', newData, broadcast=True)