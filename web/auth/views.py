from django.shortcuts import render
from django.contrib.auth.models import User

def signup(request):
    username = ""
    password = ""
    user = User.objects.create_user(username, password)
    user.save()

def login(request):
    