from fastapi import FastAPI
from pydantic import BaseModel
from app import load_html, setup_collection,retrieve,get_groq_client,load_pdf

class ChatRequest(BaseModel):
    question: str

collection=None
chat_history=[]

app = FastAPI()
def load_notes_at_startup():
    global collection
    with open(r"L1_Variables_DataTypes.html","rb") as f:
        content= f.read()

    chunks=load_html(content)
    print(f"Chunks extracted: {len(chunks)}")
    print(f"First chunk preview: {chunks[:2] if chunks else 'EMPTY'}")
    
    collection=setup_collection(chunks)
    print(f"Loaded {len(chunks)} chunks") 


def Study_assstant(question,collection):
    groq_client=get_groq_client()
    relevant_notes=retrieve(question,collection)
    context="\n".join(relevant_notes)
    chat_history.append({"role":"user","content":question})

    messages =[
        {
            "role":"system",
            "content":f"""you are a helpful studynassistant.Answer ONLY from the notes below.
            if the answer is not in the notes,say i don't have that in my notes.
        
        Notes:
        {context}"""
                }

    ]+ chat_history
    response =groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages      
    )

    answer = response.choices[0].message.content
    chat_history.append({
        "role":"assistant",
        "content":answer
    })

    return answer       
@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {"answer":Study_assstant(request.question,collection)}



load_notes_at_startup()



