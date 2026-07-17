import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing import TypedDict
from langchain_groq import ChatGroq

# Carrega a sua chave secreta do arquivo .env
load_dotenv()

# 1. Indicando qual arquivo PDF queremos abrir
carregador = PyPDFLoader("How Task Analysis Can Help Autistic and ADHD Individuals.pdf")

# 2. O Python lê o arquivo e o divide em páginas
paginas = carregador.load()

# 3. Criamos uma caixinha vazia para juntar todo o texto
contexto = ""

# 4. Passamos por cada página colando o texto dela na nossa caixinha
for pagina in paginas:
    contexto += pagina.page_content + "\n"

print("--- Sucesso! ---")
print(f"Conseguimos extrair {len(paginas)} páginas de texto.")

modelo = ChatGroq(model="llama-3.3-70b-versatile")

# --- PASSO 4: CRIANDO AS REGRAS (TEMPLATE) ---
regras_do_leitor = """Você é um assistente prestativo. Use APENAS o seguinte contexto para responder à pergunta.
Se você não souber a resposta baseada no documento, diga: 'Não sei informar com base no documento'.

Contexto retirado do PDF:
{contexto_do_pdf}

Pergunta do usuário:
{pergunta_do_usuario}
"""
molde_prompt = PromptTemplate.from_template(regras_do_leitor)


# --- PASSO 5: CRIANDO O MAPA (LANGGRAPH) ---

# 1. Criamos um "formulário" para organizar as informações que vão viajar pelo mapa
class EstadoDoPDF(TypedDict):
    pergunta_do_usuario: str
    contexto_do_pdf: str
    resposta_da_ia: str

# 2. A "estação de trabalho" que junta tudo e pede para a IA pensar
def gerar_resposta(estado: EstadoDoPDF):
    # Juntamos as regras com o modelo da Groq
    cadeia = molde_prompt | modelo 
    
    # Entregamos o texto do PDF e a pergunta para a IA
    resposta = cadeia.invoke({
        "contexto_do_pdf": estado["contexto_do_pdf"],
        "pergunta_do_usuario": estado["pergunta_do_usuario"]
    })
    
    # Guardamos a resposta final
    return {"resposta_da_ia": resposta.content}

# 3. Desenhando o mapa
mapa_pdf = StateGraph(EstadoDoPDF)
mapa_pdf.add_node("leitor", gerar_resposta)
mapa_pdf.add_edge(START, "leitor")
aplicativo_pdf = mapa_pdf.compile()

# --- PASSO 6 e 7: TESTANDO ---

print("\nEnviando a pergunta para a IA...")

# Preenchemos o formulário inicial com a pergunta e o texto do PDF
informacoes_iniciais = {
    "pergunta_do_usuario": "Qual é o assunto principal deste documento?", 
    "contexto_do_pdf": contexto # Aquela variável que guardou todo o texto no passo anterior
}

# Damos o play no nosso mapa
resultado_final = aplicativo_pdf.invoke(informacoes_iniciais)

print("\n--- Resposta da IA ---")
print(resultado_final["resposta_da_ia"])