import streamlit as st
import sqlite3
import numpy as np
import pickle
import time
from PyPDF2 import PdfReader
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory

# Define the function to update chat history
def update_chat_history(question, response):
    chat_history = st.session_state.get('chat_history', [])
    if chat_history is None:
        chat_history = []
    
    # Add the new question and response to the chat history
    chat_history.append({'question': question, 'response': response})
    st.session_state.chat_history = chat_history

# Initialize SQLite database and table
def init_db():
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT UNIQUE,
        vector BLOB
    )
    ''')
    conn.commit()
    conn.close()

# Function to store embedding in SQLite with retry logic
def store_embedding(text, vector):
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    # Serialize the vector to binary
    vector_blob = pickle.dumps(vector)
    try:
        # Try inserting the new entry
        cursor.execute('''
        INSERT INTO embeddings (text, vector) VALUES (?, ?)
        ''', (text, vector_blob))
    except sqlite3.IntegrityError:
        # If a unique constraint violation occurs, update the existing entry
        cursor.execute('''
        UPDATE embeddings SET vector = ? WHERE text = ?
        ''', (vector_blob, text))
    conn.commit()
    conn.close()

# Function to fetch all embeddings from SQLite
def fetch_all_embeddings():
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    cursor.execute('SELECT text, vector FROM embeddings')
    rows = cursor.fetchall()
    conn.close()
    embeddings = []
    for text, vector_blob in rows:
        vector = pickle.loads(vector_blob)
        embeddings.append((text, vector))
    return embeddings

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit app
st.subheader('Get Answers from Stored Vectors with LangChain & SQLite')

# Get OpenAI API key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")

# Initialize conversation memory
memory = ConversationBufferMemory()
history = ChatMessageHistory()

# Initialize SQLite database
init_db()

# PDF upload
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    text = extract_text_from_pdf(uploaded_pdf)
    st.text_area("Extracted Text", text, height=300)

    # Generate embeddings for the extracted text
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embeddings = embeddings_model.embed_documents([text])[0]  # Assuming single document

    # Store embeddings in SQLite database
    store_embedding(text, embeddings)
    st.success("PDF text has been embedded and stored in the database.")

# Query input field
query = st.text_input("Enter your query:")

if st.button("Get Answer", type="primary", use_container_width=True):
    if not query:
        st.warning("Please enter a query.")
    else:
        try:
            # Retrieve all embeddings from SQLite
            all_embeddings = fetch_all_embeddings()
            
            if all_embeddings:
                # Initialize OpenAI for Question Answering
                llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")

                # Perform similarity search
                embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
                query_embedding = embeddings_model.embed_documents([query])[0]
                
                # Find the most similar document
                similarities = [(text, np.dot(query_embedding, emb)) for text, emb in all_embeddings]
                most_similar_text = max(similarities, key=lambda x: x[1])[0]

                # Retrieve the most similar document's embedding
                most_similar_embedding = next(emb for text, emb in all_embeddings if text == most_similar_text)
                
                # Generate the response based on the most similar document
                response = chain.run(input_documents=[most_similar_embedding], question=query, max_tokens=500)
                
                # Save context to conversation memory
                memory.save_context({"question": query}, {"answer": response})
                
                # Save conversation history to session state
                update_chat_history(query, response)
                # Display response
                st.write(response)
            else:
                st.warning("No embeddings found in the database.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Initialize list to store last 10 questions
if "last_10_questions" not in st.session_state:
    st.session_state.last_10_questions = []

# Display conversation history
st.subheader("Conversation History")
if "chat_history" in st.session_state:
    for turn in st.session_state["chat_history"]:
        st.write(f"User: {turn['question']}")
        st.write(f"Bot: {turn['response']}")
    
    # Add button to clear history
    if st.button("Clear History"):
        st.session_state["chat_history"] = []
