from django.contrib.auth.decorators import login_required
from django.shortcuts import render



def home_view(request):
    '''Docstring here.'''
    arquivo = None
    mensagem = ""
    if request.method=='POST':
        arquivo = request.FILES.get('arquivo')
        mensagem = "Enviada"
    context = {
        "arquivo": arquivo,
        "mensagem": mensagem
    }

    return render(
        request,
        'home/index.html',
        context,
    )
