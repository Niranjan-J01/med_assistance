import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# reranker 

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

# retrival

class retrival:

    def __init__(self , embedding_part , vector_db , reranker):

        self.embedding_part = embedding_part
        self.vector_db = vector_db
        self.reranker = reranker
        
    def retrival_(self , query , retrival_K = 10 ,top_k =3,  score_thresold = 0.0):

        # query embedding 

        query_embbed =self.embedding_part.query_embed([query])[0]

        # simantic search 

        results = self.vector_db.collection.query(
            query_embeddings = [query_embbed.tolist()] , 
            n_results = retrival_K
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
            
            if len(final_retrival)==0 :

                return []

            pairs = [(query , doc["document"]) for doc in final_retrival]

            score = self.reranker.predict(pairs)

            for (docu , score_) in zip(final_retrival , score):

                docu["cross_score"] = float(score_)
            
            final_retrival.sort(
                key=lambda x: x["cross_score"],
                reverse=True
            )

        return final_retrival[:top_k]

retrival_pipe = retrival(embeding , vector_database , reranker)


"""for doc in results:
    print(f"\nRank: {doc['rank']}")
    print(f"Similarity: {doc['similarity_score']}")
    print(f"Content: {doc['document'][:200]}")
"""


# conetext builder 

def context_builder (results) : 

    context = []
    citaion = []

    for  doc in results:

        content = doc["document"]

        metadata = doc ["metadata"]


        SOURCE = metadata.get("source" , "UNKNOWN")
        PAGE = metadata.get("page" , "N/A")

        block = F"""
SOURCE : {SOURCE}
PAGE : {PAGE}

{content}

"""
        context.append(block)
        citaion.append(f"""
            SOURCE : {SOURCE}
            PAGE : {PAGE}
        """)

    context_part = "\n\n".join(context)

    return context_part , citaion


# prompt 

class PromptBuilder:

    def __init__(self):
        self.system_prompt = """
You are an AI Medical Assistant.

Your job is to answer ONLY using the provided medical context.

Rules:
1. Use ONLY the provided context.
2. Use the provided medical context to answer the user's question.

If the context contains partial information,
provide the best answer possible using ONLY that information.

If the context contains no relevant information,
say:

"I couldn't find sufficient information in the provided medical documents."

Never use outside knowledge.
Never fabricate information.
3. Do NOT make up facts.
4. Do NOT diagnose diseases.
5. Keep answers medically accurate and concise.
6. Mention uncertainty whenever the context is incomplete.
7. Explain in simple language whenever possible.
8. Do not mention information that is not supported by the retrieved documents.
9. When answering, cite the supporting sources using the format:

[Source: SOURCE_NAME | Page: PAGE_NUMBER]
"""

    def build_prompt(self, query, context):

        prompt = f"""
{self.system_prompt}

MEDICAL CONTEXT :

{context}

USER QUESTION :

{query}

ANSWER :

Provide a clear, evidence-based answer using only the medical context above.
"""

        return prompt

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

from groq import Groq

class Response_generater :

    def __init__(self , api = api_key , model="llama-3.3-70b-versatile"):

        self.client = Groq(api_key=api)
        self.model = model
    
    def generate (self , prompt):

        Response = self.client.chat.completions.create(

            model = self.model,

            messages=[
                {
                    "role" : "user",
                    "content": prompt
                }
            ],
            temperature= 0.0,
            max_completion_tokens=1024
            
        )
        return {
            "answer": Response.choices[0].message.content,
            "model": self.model,
            "usage": Response.usage
        }

class ConfidenceEstimator:

    def __init__(
        self,
        high_threshold=0.80,
        medium_threshold=0.60
    ):

        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def estimate(self, retrieved_docs, answer):

        if len(retrieved_docs) == 0:

            return {
                "score": 0.0,
                "level": "Low",
                "reason": "No relevant documents were retrieved."
            }

        cross_scores = [
            doc["cross_score"]
            for doc in retrieved_docs
        ]

        avg_cross = sum(cross_scores) / len(cross_scores)

        confidence = 1 / (1 + pow(2.71828, -avg_cross))

        if "couldn't find sufficient information" in answer.lower():

            confidence *= 0.60

        if confidence >= self.high_threshold:

            level = "High"

        elif confidence >= self.medium_threshold:

            level = "Medium"

        else:

            level = "Low"

        return {

            "score": round(confidence, 2),

            "level": level,

            "reason": f"Average CrossEncoder score = {avg_cross:.2f}"

        }

confidence =  ConfidenceEstimator()  

class safetylayer:

    def __init__(self):

        self.emergency_keywords = [
            "chest pain",
            "can't breathe",
            "cannot breathe",
            "difficulty breathing",
            "severe bleeding",
            "unconscious",
            "seizure",
            "suicidal",
            "kill myself",
            "sudden severe headache",
            "face drooping",
            "slurred speech"
        ]

        self.unsafe_keywords = [
            "diagnose me",
            "give me a diagnosis",
            "how much should i take",
            "what dosage should i take",
            "prescribe me",
            "which prescription drug should i take",
            "should i stop my medication",
            "can i stop taking my medication",
            "change my medication dose",
            "increase my medication dose"
        ]

    def contain_keywords( self , query , keywords ):

        query = query.lower()

        return any( keyword in query for keyword in keywords)
    
    def check_point(self , query):

        # emergency check point 

        if self.contain_keywords(query , self.emergency_keywords) :

            return {
                "allowed" : False,
                "type" : "emergency",
                "message" : (
                    "Your message may describe a medical emergency. "
                    "Please seek immediate medical attention or contact "
                    "local emergency services."
                )
            }

        # unnsafe_keyword 

        if self.contain_keywords(query , self.unsafe_keywords) :

             return {
                "allowed": False,
                "type": "unsafe_request",
                "message": (
                    "I can't provide personalized diagnoses, prescribe "
                    "medication, recommend specific dosages, or advise you "
                    "to stop or change prescribed treatment. Please consult "
                    "a qualified healthcare professional."
                )
            }
        
        return {

            "allowed" : True,
            "type" : "Normal",
            "message" : None
        }

safety = safetylayer()

while True:

    query = input("Ask: ")

    safety_result = safety.check_point(query)

    if not safety_result["allowed"]:
        print(safety_result["message"])

    else:

        results = retrival_pipe.retrival_(
            query,
            score_thresold=0.35
        )

        context, citations = context_builder(results)

        prompt_builder = PromptBuilder()

        prompt = prompt_builder.build_prompt(
            query,
            context
        )

        llm = Response_generater()

        response = llm.generate(prompt)

        estimator = confidence.estimate(results , response["answer"])

        print(f""" answer : 
            
        {response["answer"]}

        CONFIDENCE :
                
        score : {estimator["score"]}
        level : {estimator["level"]}
        reason : {estimator["reason"]}

        CITATION :
        """)

        refusal = [
            "couldn't find sufficient information",
            "insufficient information",
            "not enough information",
            "unable to answer from the provided context"
        ]

        is_refusal = any(
            phrase in response["answer"].lower()
            for phrase in refusal
        )

        if not is_refusal:
            print("ENTERED CITATION IF BLOCK")
            for c in citations:
                print(c)
        else :
            print("N/A")            

        print(
            "This information is for educational purposes only "
            "and does not replace professional medical advice."
        )
            











