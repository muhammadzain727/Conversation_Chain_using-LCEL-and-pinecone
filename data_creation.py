import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
api=os.environ.get("PINECONE_API_KEY")
PINECONE_API_KEY = Pinecone(api_key=api)
def vector_insertion(docs,embeddings,index_name):

    doc_search=PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,  
        
        index_name=index_name,
        )
    return doc_search
def text_splitter(file_path):
    splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    loader=PyMuPDFLoader(file_path)
    data=loader.load()
    texts = splitter.split_documents(data)
    return texts
folder_path='file_path'
embed = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
doc_search_list=[]
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):  
        file_path = os.path.join(folder_path, filename)
        docs=text_splitter(file_path)
        embeddings= embed
        index_name="vectordbtwo"
        vector_insertion(docs,embeddings,index_name)
