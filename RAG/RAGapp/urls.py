from django.urls import path
from . import views

urlpatterns = [
    path('', views.question_answer, name='home'),
    path('ajax/', views.ajax_response, name='ajax_response'),
]
