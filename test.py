from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
api=os.environ.get("PINECONE_API_KEY")
PINECONE_API_KEY = Pinecone(api_key=api)
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
index_name="vectordbtwo"
docsearch = PineconeVectorStore(embedding=embeddings, index_name=index_name)
input_query="The first three parts laid the foundation for this concept of tiny, daily actions?"
num_chunks=10
retriever=docsearch.as_retriever(search_type='similarity', search_kwargs={'k': num_chunks,'filter': {'source':'your file path'}})


def qa_ret(retriever,input_query):
    try:
        template = """You are AI assistant that assisant user by providing answer to the question of user by extracting information from provided context:
        {context} and chat_history if user question is related to chat_history take chat history as context .
        if you donot find any relevant information from context for given question just say ask me another quuestion. you are ai assistant.
        Answer should not be greater than 3 lines.
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        setup_and_retrieval = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
                )
            # Load QA Chain
        GOOGLE_API_KEY ="<your-api-key>"
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key = GOOGLE_API_KEY)
        output_parser= StrOutputParser()
        rag_chain = (
        setup_and_retrieval
        | prompt
        | model
        | output_parser
        )
        respone=rag_chain.invoke(input_query)
        return respone
    except Exception as ex:
        return ex
    
results=qa_ret(retriever,input_query)
print(results)