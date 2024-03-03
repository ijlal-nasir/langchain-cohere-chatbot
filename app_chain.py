from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.llms import cohere
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY=os.getenv('COHERE_API_KEY')

def get_chatbot_chain():
    # 1. Load knowledge base
    loader = CSVLoader(file_path="faq.csv")
    documents = loader.load()

    # 2. Transform knowledge base into embeddings
    embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

    # 3. Store embeddings in a vector database
    vectorstore = faiss.FAISS.from_documents(documents, embeddings_model)

    # 4. Choose LLM
    llm_model = cohere.Cohere(cohere_api_key=COHERE_API_KEY)

    # 5. Chat History & memory management
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 6. Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=vectorstore.as_retriever(), memory=memory)

    return chain
  


