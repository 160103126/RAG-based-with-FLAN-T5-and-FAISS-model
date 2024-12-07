from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Step 1: Load pre-trained model (FLAN-T5-base)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Step 2: Load FAISS index and passages
faiss_index = faiss.read_index("./data/faiss_index.bin")
passages = np.load("./data/passages.npy", allow_pickle=True)

# Step 3: Load SentenceTransformer for query embedding
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Define retrieval and generation functions
def retrieve(query, top_k=5):
    query_embedding = embedder.encode([query])
    _, indices = faiss_index.search(query_embedding, top_k)
    return [passages[i] for i in indices[0]]

def generate_answer(query, contexts):
    context_text = " ".join(contexts)
    input_text = f"question: {query} context: {context_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 5: Streamlit App for Interactive Q&A
st.title("RAG-based Q&A with FLAN-T5 and FAISS")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:")

if query:
    contexts = retrieve(query)  # Retrieve relevant contexts
    answer = generate_answer(query, contexts)  # Generate answer
    st.session_state.history.append({"query": query, "answer": answer})

# Display conversation history
for qa in st.session_state.history:
    st.write(f"**Q:** {qa['query']}")
    st.write(f"**A:** {qa['answer']}")
