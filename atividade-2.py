import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 2. Inicializando o modelo que você escolheu:
modelo = ChatGroq(model="llama-3.3-70b-versatile")

from langchain_core.messages import SystemMessage, HumanMessage

# Criando as instruções
mensagens = [
    SystemMessage(content="Traduza o seguinte texto de Inglês para Português"),
    HumanMessage(content="Hello, how are you?")
]

# Enviando para o modelo Groq
resposta = modelo.invoke(mensagens)

# Mostrando o resultado na tela
print("--- Resposta da Tradução Simples ---")
print(resposta.content)

from langchain_core.prompts import ChatPromptTemplate

# Criando o formulário com espaços em branco
system_template = "Traduza o seguinte texto de Inglês para {idioma}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template), 
    ("user", "{text}")
])

# Aqui nós juntamos o formulário com o modelo da Groq
cadeia_de_traducao = prompt_template | modelo

# Teste 1: Traduzindo para o Italiano
resultado_italiano = cadeia_de_traducao.invoke({
    "idioma": "Italiano", 
    "text": "Hello, how are you?"
})

print("\n--- Tradução para Italiano ---")
print(resultado_italiano.content)

# Teste 2: Traduzindo para o Espanhol (com um texto novo)
resultado_espanhol = cadeia_de_traducao.invoke({
    "idioma": "Espanhol", 
    "text": "I am learning Python and it is fun!"
})

print("\n--- Tradução para Espanhol ---")
print(resultado_espanhol.content)