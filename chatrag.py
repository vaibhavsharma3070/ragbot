import os
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Initialize Pinecone and OpenAI
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def ingest_document(file_path):
    # Load and split the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create or get Pinecone index
    index_name = "rag-chatbot"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)  # OpenAI embedding dimension
    index = pinecone.Index(index_name)
    
    # Create vector store
    vectorstore = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    return vectorstore

def create_chatbot(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    return qa_chain

def chat(qa_chain, query, chat_history=[]):
    result = qa_chain({"question": query, "chat_history": chat_history})
    return result["answer"], result["source_documents"]

# Example usage
file_path = "path/to/your/document.pdf"
vectorstore = ingest_document(file_path)
chatbot = create_chatbot(vectorstore)

query = "What was UK&I Rental profit in FY2024?"
answer, sources = chat(chatbot, query)
print(f"Answer: {answer}")
print("Sources:")
for source in sources:
    print(f"- {source.page_content[:100]}...")