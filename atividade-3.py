import os
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

# 1. Configurando o acesso ao modelo Groq
load_dotenv()
modelo = ChatGroq(model="llama-3.3-70b-versatile")

# 2. Criando a caixinha que vai guardar a memória
memoria = MemorySaver()

# 3. Função que faz a IA ler e responder as mensagens
def chamar_modelo(estado: MessagesState):
    resposta_do_modelo = modelo.invoke(estado["messages"])
    # Devolve a resposta pronta para ser guardada
    return {"messages": [resposta_do_modelo]}

# 4. Construindo o nosso mapa (Grafo)
mapa = StateGraph(MessagesState)

# Adicionando uma "parada" no mapa chamada "chatbot"
mapa.add_node("chatbot", chamar_modelo)

# Criando o caminho: o mapa começa (START) e vai para o "chatbot"
mapa.add_edge(START, "chatbot")

# 5. Juntando o mapa pronto com a nossa memória
chat_com_memoria = mapa.compile(checkpointer=memoria)

# 5. Criando a "identidade" da nossa conversa (Thread ID)
configuracao = {"configurable": {"thread_id": "minha_primeira_conversa"}}

# 6. Primeira interação: Vamos nos apresentar
print("Enviando a primeira mensagem...")
mensagem_1 = {"messages": [("user", "Olá, eu sou o Felipe!")]}

# Mandando para o mapa (que agora tem memória)
resposta_1 = chat_com_memoria.invoke(mensagem_1, config=configuracao)

print("--- Resposta da IA ---")
# O [-1] serve para pegar apenas a última mensagem (a resposta)
print(resposta_1["messages"][-1].content)


# 7. Segunda interação: Testando a memória
print("\nEnviando a segunda mensagem...")
mensagem_2 = {"messages": [("user", "Como eu me chamo?")]}

resposta_2 = chat_com_memoria.invoke(mensagem_2, config=configuracao)

print("--- Resposta da IA ---")
print(resposta_2["messages"][-1].content)