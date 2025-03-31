from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

llm = Groq(model="llama3-70b-8192",api_key=os.environ.get("GROQ_API_KEY"))
embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")

Settings.llm = llm
Settings.embed_model = embed_model


de_tools_blog = SimpleDirectoryReader("./",required_exts=[".pdf", ".docx"]).load_data()
index = VectorStoreIndex.from_documents(de_tools_blog)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("How many tools are there?")
print(response)
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = CondensePlusContextChatEngine.from_defaults(    
   index.as_retriever(),    
   memory=memory,    
   llm=llm
)

response = chat_engine.chat(    
   "What tools are suitable for data processing?"
)
print(str(response))

response = chat_engine.chat(
    "Can you create a diagram of a data pipeline using these tools?"
)
print(str(response))