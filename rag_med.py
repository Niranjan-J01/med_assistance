import os 
os.environ["CUDA_VISIBLE_DEVICEs"] = "0"

import torch as ts 
import pandas as pd
ts.cuda.get_device_name(0)

# loaders
medquad = pd.read_csv("medquad.csv")

med_dict = medquad.to_dict(orient="records")
#print(med_dict[:1])

from langchain_core.documents import Document  

extra_docs = [
    Document(
        page_content=f"Q: {row['question']}\nA: {row['answer']}",
        metadata={"source": row['source'], "focus_area": row['focus_area']}
    )
    for row in med_dict
]


from langchain_community.document_loaders import DirectoryLoader , PyMuPDFLoader

loader = DirectoryLoader(
    path=r".\data",
    glob=["*.pdf", "*.csv"],
    loader_cls=PyMuPDFLoader
)
doc = loader.load()

# chunking 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_splitter(x):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )
    doc = text_splitter.split_documents(x)
    return doc

chunks_  = text_splitter(doc)
chunks = chunks_ + extra_docs

# embedding

from sentence_transformers import SentenceTransformer

class embed():

    def __init__(self , model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name , device="cuda")
        print(f"activating model : {self.model_name} dimension = {self.model.get_embedding_dimension()}")
    
    def generate_embedding(self, chunks):

        docs = [text.page_content for text in chunks]
        text  = self.model.encode( docs , show_progress_bar = True , batch_size = 32)
        print(f"embedding shape : {text.shape}")
        return text

    def query_embed(self , query):

        text = self.model.encode(query)

        return text

embeding = embed()

# vector store 

import chromadb 
import uuid

class vector_DB:

    def __init__(self ,collection_name = "med_pdfs", path = r"C:\Users\njnin\OneDrive\Desktop\task 2\vector_DB_med"):
        
        self.presistent_directory = path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        self._intiallization_()
    
    def _intiallization_(self):

        os.makedirs(self.presistent_directory , exist_ok=True)

        self.client = chromadb.PersistentClient(path = self.presistent_directory)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata={"discription" : "this a vectordatabase to store embbededd data"}
        )
        print(f"doc count : {self.collection.count()}")

    def add_document(self , document  , embbeddings , batch = 5000):
        
        for start in range(0 , len(document) , batch):
            end = start + batch 
            batch_emd = embbeddings[start:end]
            batch_doc = document[start:end]
            id = []
            net_metadata = []
            net_embedding = []
            docs = []

            for i , (doc,emb) in enumerate(zip(batch_doc,batch_emd)):

                doc_id = f"doc_id {uuid.uuid1()}"
                id.append(doc_id)

                metadata = dict(doc.metadata)
                metadata["index"] = i
                metadata["content lenght"] = len(doc.page_content)
                
                net_metadata.append(metadata)
                net_embedding.append(emb)

                docs.append(doc.page_content)
            
            self.collection.add(
                ids = id,
                metadatas = net_metadata,
                embeddings = net_embedding,
                documents = docs
            )

        return self.collection 

emd_data = embeding.generate_embedding(chunks)

vector_database = vector_DB()
if vector_database.collection.count() == 0:
    vector_database.add_document(chunks , emd_data)
else:
    print(f"DB already has {vector_database.collection.count()} docs!")

# retrival

class retrival:

    def __init__(self , embedding_part , vector_db ):

        self.embedding_part = embedding_part
        self.vector_db = vector_db
    
    def retrival_(self , query , top_K = 5 , score_thresold = 0.0):

        # query embedding 

        query_embbed =self.embedding_part.query_embed([query])[0]

        # simantic search 

        results = self.vector_db.collection.query(
            query_embeddings = [query_embbed.tolist()] , 
            n_results = top_K
        )

        final_retrival = []

        if results["documents"] and results["documents"][0]:

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadata = results["metadatas"][0]
            distance = results["distances"][0]
        
        for i , (id , doc , meta , dist) in enumerate(zip(ids , documents , metadata , distance)) :

            similarity  = 1/(1+ dist)

            if similarity > score_thresold :

                retrived = {
                    "id" : id,
                    "document":doc,
                    "metadata":meta , 
                    "distance" : dist,
                    "similarity_score" : similarity,
                    "rank":1+i
                }

                final_retrival.append(retrived)

        return final_retrival

retrival_pipe = retrival(embeding , vector_database)

results = retrival_pipe.retrival_(
    "How to prevent Glaucoma ?",
     score_thresold=0.35
)

for doc in results:
    print(f"\nRank: {doc['rank']}")
    print(f"Similarity: {doc['similarity_score']}")
    print(f"Content: {doc['document'][:200]}")
#

