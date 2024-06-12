from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
api=os.environ.get("PINECONE_API_KEY")
PINECONE_API_KEY = Pinecone(api_key=api)
def vector_db_creation(index_name):
        PINECONE_API_KEY.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        )
index_name = "vectordbtwo"
try:
    vector_db_creation(index_name)
    print("DB created")
except Exception as e:
      print(e)