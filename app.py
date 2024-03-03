from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.llms import cohere
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

COHERE_API_KEY="X469gZpKPQRC3OR9Jn9kfPGf65JMHG7xzVYT08fn"

#------- Steps ------#
# 1. Load Knowledge Base
# 2. Transform knowledge base documents to embeddings
# 3. Store embeddings in a vector database
# 4. Pass relevant information to the LLM 
# 5. LLM will prepare the answer
# 6. Chat history and Memory Management


# 1. Load knowledge base
loader = CSVLoader(file_path="faq.csv")
documents = loader.load()
# print(documents)

# 2. Transform knowledge base into embeddings
embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

# 3. Store embeddings in a vector database
vectorstore = faiss.FAISS.from_documents(documents, embeddings_model)
results = vectorstore.similarity_search('Do you have a free plan?')
# print(results)
# embeddings = embeddings_model.embed_documents(['hello', 'world' ])

# 4. Choose LLM
llm_model = cohere.Cohere(cohere_api_key=COHERE_API_KEY)

# 6. Chat History & memory management
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5. Create conversation chain
# 5.2 Pass relevant information to the LLM 
# 5.3 LLM will prepare the answer
chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=vectorstore.as_retriever(), memory=memory)





# Test
query = "Do you have a team plan?" 
response = chain.invoke({"question": query})
print(response)

