import os 
os.environ["CUDA_VISIBLE_DEVICEs"] = "0"

import torch as ts 
ts.cuda.get_device_name(0)

# loader

from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
    path=r"/data",
    glob=["*.pdf", "*.csv"],
    loader_cls=PyPDFLoader
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

chunks  = text_splitter(doc)

# embedding

from sentence_transformers import SentenceTransformer

class embed(SentenceTransformer):

    def __init__(self , model_name = "all-MiniLM-L6-v2"):
        self.model = model_name
        self.emd = SentenceTransformer(self.model , device="cuda")
        print(f"activating model : {self.model} dimension = {self.model.get_embedding_dimension()}")
    
    def get_embedding(self, chunks):
        text  = self.model.encode( chunks , show_progress_bar = True , batch_size = 32)
        print(f"embedding shape : {text.shape()}")
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

        os.makedirs(self.presistent_dictionary , exist_ok=True)

        self.client = chromadb.PersistentClient(path = self.presistent_directory)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata={"discription" : "this a vectordatabase to store embbededd data"}
        )
        print(f"doc count : {self.collection}")

    def add_document(self , document  , embbeddings):

        id = []
        net_metadata = []
        net_embbedding = []
        documents = []

        for i , (doc,emb) in enumerate(zip(document , embbeddings)):

            doc_id = f"doc_id {uuid.uuid1()}"
            id.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["index"] = i
            metadata["content lenght"] = len(doc.page_content)
            
            net_metadata.append(metadata)
            net_embbedding.append(emb)

            documents.append(doc.page_content)
        
        self.collection(
            ids = id,
            metadatas = net_embbedding,
            embbeddings = net_embbedding,
            docs  = documents
        )

        return self.collection




        

