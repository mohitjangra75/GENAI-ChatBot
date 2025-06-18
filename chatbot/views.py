import json, os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

corpus_embeddings = None
chunk_metas = []

def chunk_text(text, max_words=250, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks

def load_documents_and_index():
    global corpus_embeddings, chunk_metas
    if corpus_embeddings is not None:
        return

    file_path = os.path.join(settings.BASE_DIR, "data", "murder_cases.json")
    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus_chunks = []
    for i, doc in enumerate(documents):
        chunks = chunk_text(doc['full_text'])  
        for chunk in chunks:
            corpus_chunks.append(chunk)
            chunk_metas.append({
                "doc_id": i,
                "chunk_text": chunk,
                "title": doc["title"],
                "url": doc["url"]
            })

    corpus_embeddings = embedder.encode(corpus_chunks, convert_to_numpy=True)

def search_similar_chunks(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunk_metas[idx] for idx in top_indices]

def generate_answer(query, retrieved_chunks, max_length=768):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"[Reference {i+1}]\n"
            f"Title: {chunk['title']}\n"
            f"URL: {chunk['url']}\n"
            f"Text: {chunk['chunk_text']}\n\n"
        )

    prompt = (
        "You are a highly knowledgeable legal assistant. Based ONLY on the legal context below, "
        "provide a detailed, clear, and formal explanation answering the user's question. "
        "Cite relevant laws explicitly by section, provide examples if applicable, "
        "and always reference the information with [Reference 1], [Reference 2], etc. "
        "Include the URL for each reference in your answer in parentheses immediately after the reference tag. "
        "Avoid vague or overly brief answers.\n\n"
        f"Context:\n{context}\n"
        f"User Question: {query}\n"
        "Detailed Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
        temperature=0.7,
        length_penalty=1.2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@csrf_exempt
def index(request):
    chat_history = request.session.get('chat_history', [])
    answer = None
    question = ""

    if request.method == "POST":
        question = request.POST.get('question', '').strip()
        if question:
            load_documents_and_index()
            retrieved = search_similar_chunks(question)
            answer = generate_answer(question, retrieved)
            chat_history.append({"question": question, "answer": answer})
            request.session['chat_history'] = chat_history

    return render(request, "chatbot/index.html", {
        "answer": answer,
        "question": question,
        "chat_history": chat_history
    })
