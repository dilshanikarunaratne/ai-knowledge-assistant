import ollama

with open("data/company_notes.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

question = input("Ask a question about the company: ")

prompt = f"""
You are an AI knowledge assistant.
Answer the user's question using only the information below.

Knowledge:
{knowledge}

Question:
{question}
"""

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nAnswer:")
print(response["message"]["content"])