from django.urls import path

from . import views

app_name = "auth"
urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path("login/", views.login, name="login"),
]