
from django.urls import path
from .views import home_view, teste_treinado_view

urlpatterns = [
    path('', home_view, name='home'),
    path('teste_treinado/', teste_treinado_view, name='teste_treinado_view'),
    
]

