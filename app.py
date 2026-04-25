import ollama

def split_text(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def find_relevant_chunks(question, chunks):
    question_words = set(question.lower().split())

    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:2]]

with open("data/company_notes.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

chunks = split_text(knowledge)

question = input("Ask a question about the company: ")

relevant_chunks = find_relevant_chunks(question, chunks)

context = "\n\n".join(relevant_chunks)

prompt = f"""
You are an AI knowledge assistant.
Answer the question using only this context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I don't know based on the provided knowledge."
"""

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nAnswer:")
print(response["message"]["content"])