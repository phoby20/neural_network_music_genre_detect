from django.urls import path
from .views import *
from django.conf.urls import url

app_name = 'index'

urlpatterns = [
    path('', homeView, name='index'),

]