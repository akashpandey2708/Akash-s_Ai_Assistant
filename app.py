import os
import gradio as gr
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
MODEL = "openai/gpt-oss-20b:free"  # you can switch to "mistral", "llama-3-8b-instruct", etc.
db_name = "vector_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
API_KEY = os.getenv("API_KEY") 

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


# ---------------- VECTORSTORE ----------------
def initialize_vectorstore():
    if os.path.exists(db_name) and os.listdir(db_name):
        print("Vectorstore already exists. Loading...")
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    else:
        print("No existing vectorstore found. Creating new one...")
        loader = PyPDFLoader("resume.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_name,
        )
        
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore


# ---------------- OPENROUTER WRAPPER ----------------
class OpenRouterLLM(LLM):
    def _call(self, prompt, stop=None):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model": MODEL}

    @property
    def _llm_type(self):
        return "openrouter"


# ---------------- CONVERSATION CHAIN ----------------
def initialize_conversation_chain(vectorstore):
    llm = OpenRouterLLM()

    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Akash Kumar Pandey's helpful AI assistant. Answer clearly and confidently as his assistant. Do not say things like 'According to the context' or 'Based on the document'."),
        ("human", "Here is some background information:\n{context}\n\nNow, answer this question:\n{question}")
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )


# ---------------- CHAT FUNCTION ----------------
def chat(message, history):
    result = conversation_chain.invoke({"question": message, "chat_history": history})
    return result.get("answer") or result.get("result")


# ---------------- UI ----------------
if __name__ == "__main__":
    vectorstore = initialize_vectorstore()
    conversation_chain = initialize_conversation_chain(vectorstore)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 Akash's AI Assistant  
            Ask me anything based on Akash's profession.  
            I’ll answer clearly and confidently as his assistant.  
            """
        )

        with gr.Row():
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    label="Chat with Akash's Assistant",
                    type="messages",
                    height=500,
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"),
                )
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        scale=9,
                        lines=1,
                        show_label=False,
                    )
                    send = gr.Button("Send", scale=1, min_width=80)
                clear = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Instructions")
                gr.Markdown(
                    """
                    - Ask questions about Akash's resume.  
                    - My knowledge bas is limited to Akash's career.  
                    - The assistant remembers the chat.  
                    - Use *Clear Chat* to reset conversation.  
                    """
                )

        def user_message(user_input, history):
            response = chat(user_input, history)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            return "", history

        send.click(user_message, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch()

