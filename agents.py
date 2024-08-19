import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = 'place_your_api_here'  # replace with your openai api key

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
print(retriever)

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
print(wiki.name)

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")
print(retriever_tool.name)

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

tools = [wiki, arxiv, retriever_tool]
print(tools)

from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent",api_key='place_your _api_here')# replace with your openai api key
print(prompt.messages)

from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor)

print(agent_executor.invoke({"input": "Tell me about langsmith."})) #you can give any query
print(agent_executor.invoke({"input": "what is artificial intelligence?"})) #you can give any query
