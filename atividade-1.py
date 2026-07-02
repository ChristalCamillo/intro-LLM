import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Você é um expert em história dos LLMs."},
        {"role": "user", "content": "Conte uma história sobre o desenvolvimento da Inteligência Artificial até a invenção dos LLMs."}
    ],
    temperature=0.1
)
print(completion.choices[0].message.content)