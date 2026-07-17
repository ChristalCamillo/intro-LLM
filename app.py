import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Carrega a sua chave de segurança
load_dotenv()

# Configuração da interface
st.title("Chat com IA usando LangChain")

# Cria a caixa de texto e guarda o que foi digitado na variável 'prompt'
prompt = st.text_area("Digite sua pergunta:")

# 1ª Condição: O usuário clicou no botão?
if st.button("Enviar"):
    
    # 2ª Condição: A caixa de texto tem alguma coisa escrita?
    if prompt:
        # Se a caixa NÃO estiver vazia, liga a IA
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        
        # Envia a pergunta para a IA
        resposta = llm.invoke(prompt)
        
        # Mostra a resposta na tela
        st.write("**Resposta:**", resposta.content)
        
    else:
        # Se a caixa ESTIVER vazia, mostra um aviso na tela
        st.warning("Digite uma pergunta antes de enviar.")