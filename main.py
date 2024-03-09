import os
import pickle
import pandas as pd
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

def storeDocEmbeds(file, filename):
    """
    Stores the document embeddings in a pickle file.
    
    Parameters:
    - file: The document file content to embed.
    - filename: The filename for the pickle file to store embeddings.
    """
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(file)
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    os.remove(tmp_file_path)

    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)

def getDocEmbeds(file, filename):
    """
    Retrieves or creates document embeddings stored in a pickle file.
    
    Parameters:
    - file: The document file content to embed.
    - filename: The filename for the pickle file.
    
    Returns:
    - vectors: The document embeddings.
    """
    if not os.path.isfile(filename + ".pkl"):
        storeDocEmbeds(file, filename)
    
    with open(filename + ".pkl", "rb") as f:
        vectors = pickle.load(f)
        
    return vectors

def conversational_chat(query, chain, history):
    """
    Generates a conversational response using the provided chat model.
    
    Parameters:
    - query: User's input query.
    - chain: The conversational retrieval chain object.
    - history: The conversation history.
    
    Returns:
    - result['answer']: The generated response to the query.
    """
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    print("Log: ")
    print(history)
    return result["answer"]

def setup_chain(uploaded_file, user_api_key, MODEL='gpt-4-0125-preview'):
    """
    Sets up the conversational chain for chat interactions.
    
    Parameters:
    - uploaded_file: The uploaded file object from Streamlit.
    - user_api_key: The API key for OpenAI.
    - MODEL: The model identifier for the chat model.
    
    Returns:
    - chain: The initialized conversational retrieval chain.
    """
    os.environ["OPENAI_API_KEY"] = user_api_key
    uploaded_file.seek(0)
    file = uploaded_file.read()

    vectors = getDocEmbeds(file, uploaded_file.name)
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0, model_name=MODEL, streaming=True),
                                                  retriever=vectors.as_retriever(), chain_type="stuff")
    return chain
