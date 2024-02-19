# papaya_quality
Avaliação de quailidade com inteligência artificial em mamões 

Este é um projeto para a disciplina de Tópicos Especiais em Engenharia de Software
UFERSA 2023.2

## Como Rodar o Projeto

Siga estas etapas para configurar e executar o projeto localmente.

### Pré-requisitos

Certifique-se de ter instalado em seu ambiente de desenvolvimento:

- Python (versão 3.10.9)
- pip (gerenciador de pacotes do Python)
- Virtualenv (opcional, mas altamente recomendado)

```bash
# Clone o repositório:
git clone https://github.com/LucasMatheus12/papaya_quality

# Navegue até o diretório do projeto:
cd papaya_quality

# Crie e ative um ambiente virtual (opcional, mas recomendado):
virtualenv venv
source venv/bin/activate

# Instale as dependências do projeto:
pip install -r requirements.txt

# Configure as variáveis de ambiente:
# Crie um arquivo `.env` na raiz do projeto e defina as variáveis de ambiente necessárias, como chaves secretas, configurações de banco de dados, etc.
# As variaveis de desenvolvimento estão no arquivo .env_auxiliar, pode copiar e colar dentro do arquivo .env

# Aplique as migrações do Django:
python manage.py migrate

# Crie um superusuário (opcional):
python manage.py createsuperuser
```

## Executando o Servidor de Desenvolvimento
### Execute o servidor de desenvolvimento com o seguinte comando:
```bash
python manage.py runserver
```
Acesse o projeto em seu navegador em http://localhost:8000.

### Contribuindo
Se você deseja contribuir com este projeto, siga estas etapas:

### Referências:
#1 documentação kagle
https://www.kaggle.com/code/rss1011/surface-crack-detection-cnn

### Licença
Este projeto está licenciado sob a Licença MIT.