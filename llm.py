import hashlib
import os
from typing import Optional, Iterable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.indexes import SQLRecordManager
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, RePhraseQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredWordDocumentLoader, \
    UnstructuredMarkdownLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import logging
import shutil

# 这行代码配置了日志记录的基本设置。它调用 logging.basicConfig()，这会对日志记录进行基本配置，例如设置日志记录格式、日志文件等。
# 这里没有提供具体参数，所以使用默认配置，这通常包括在控制台输出日志消息。
logging.basicConfig()

# 这行代码获取名为 "langchain.retrievers.multi_query" 的日志记录器，并将其日志级别设置为 INFO。
# 这样，任何由这个记录器产生的 INFO 级别及以上的日志消息（INFO、WARNING、ERROR、CRITICAL）都会被输出。
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 这行代码获取名为 "langchain.retrievers.re_phraser" 的日志记录器，并将其日志级别设置为 INFO。
# 同样，任何由这个记录器产生的 INFO 级别及以上的日志消息都会被输出。
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

# 加载.env文件中的环境变量
load_dotenv()

# 设置知识库 向量模型 重排序模型的路径
KNOWLEDGE_DIR = './chroma/knowledge/'
embedding_model = './BAAI/bge-large-zh-v1.5'
rerank_model = './BAAI/bge-reranker-large'
model_kwargs = {'device': 'cpu'}

# 知识库问答指令
qa_system_prompt = (
    "你叫小文，一个帮助人们解答各种问题的助手。 "
    "使用检索到的上下文来回答问题。如果你不知道答案，就说你不知道。 "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# 正常聊天指令
normal_system_prompt = (
    "你叫小文，一个帮助人们解答各种问题的助手。"
)

normal_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", normal_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    db = Chroma(collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=os.path.join('./chroma', collection_name))

    # https://python.langchain.com/v0.2/docs/how_to/indexing/
    record_manager = SQLRecordManager(
        f"chromadb/{collection_name}", db_url="sqlite:///record_manager_cache.sql"
    )
    print('record_manager: ', record_manager)
    record_manager.create_schema()
    print('record_manager: ', record_manager)
    print('record_manager.create_schema: ', record_manager.create_schema())
    documents = loader.load()
    print('documents: ', documents)

    r = index(documents, record_manager, db, cleanup="full", source_id_key="source")
    print('r: ', r)
    '''混合检索，将稀疏检索器（如BM25）与密集检索器（如嵌入相似性）相结合。
    稀疏检索器擅长根据关键字查找相关文档，而密集检索器擅长根据语义相似性查找相关文档。'''
    ensemble_retriever = EnsembleRetriever(
        retrievers=[db.as_retriever(search_kwargs={"k": 3}), BM25Retriever.from_documents(documents)]
    )
    print('ensemble_retriever: ', ensemble_retriever)

    return ensemble_retriever


def get_md5(input_string):
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()

    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))

    # 获取十六进制的哈希值
    return hash_md5.hexdigest()


def streaming_parse(chunks: Iterable[AIMessageChunk]):
    for chunk in chunks:
        yield AddableDict({'answer': chunk.content})


class MyCustomLoader(BaseLoader):
    # 支持加载的文件类型
    file_type = {
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {})
    }

    # 初始化方法  将加载的文件进行切分
    def __init__(self, file_path: str):
        loader_class, params = self.file_type[detect_filetype(file_path)]
        print('loader_class:', loader_class)
        print('params:', params)
        self.loader: BaseLoader = loader_class(file_path, **params)
        print('self.loader:', self.loader)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def lazy_load(self):
        # 懒惰切分加载
        return self.loader.load_and_split(self.text_splitter)

    def load(self):
        # 加载
        return self.lazy_load()


class MyKnowledge:
    # 向量化模型
    __embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)
    print('__embeddings:', __embeddings)

    __retrievers = {}
    __llm = OpenAI(temperature=0)

    def upload_knowledge(self, temp_file):
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        # 如果文件不存在就copy
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(temp_file, file_path)

        import gradio as gr
        return None, gr.update(choices=self.load_knowledge())

    def load_knowledge(self):
        # exist_ok=True目标目录已存在的情况下不会抛出异常。
        # 这意味着如果目录已经存在，os.makedirs不会做任何事情，也不会报错
        os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)

        # 知识库默认为空
        collections = [None]
        print('os.listdir(KNOWLEDGE_DIR):', os.listdir(KNOWLEDGE_DIR))

        for file in os.listdir(KNOWLEDGE_DIR):
            # 将知识库进行添加
            collections.append(file)

            # 得到知识库的路径
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            print('file_path:', file_path)

            # 知识库文件名进行md5编码
            collection_name = get_md5(file)
            print('collection_name:', collection_name)

            print('self.__retrievers:', self.__retrievers)
            if collection_name in self.__retrievers:
                continue
            # 创建对应加载器
            loader = MyCustomLoader(file_path)
            print('loader:', loader)
            self.__retrievers[collection_name] = create_indexes(collection_name, loader, self.__embeddings)
            print('collections:', collections)
        return collections

    def get_retrievers(self, collection):
        collection_name = get_md5(collection)
        print('知识库名字md5:', collection_name)
        if collection_name not in self.__retrievers:
            print('self.__retrievers:', self.__retrievers)
            print('True')
            return None

        retriever = self.__retrievers[collection_name]
        print('get_retrievers中:', retriever)
        ''' LLMChainFilter:过滤，对寻回的文本进行过滤。它的主要目的是根据一定的条件或规则筛选和过滤文本内容。'''
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(self.__llm),
            # https://python.langchain.com/v0.2/docs/integrations/retrievers/re_phrase/#setting-up
            # 提取问题关键元素
            base_retriever=RePhraseQueryRetriever.from_llm(retriever, self.__llm)
        )

        '''rerank https://python.langchain.com/v0.2/docs/integrations/document_transformers/cross_encoder_reranker/'''
        model = HuggingFaceCrossEncoder(model_name=rerank_model, model_kwargs=model_kwargs)
        compressor = CrossEncoderReranker(model=model, top_n=3)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=compression_retriever
        )

        print('compression_retriever:', compression_retriever)

        return compression_retriever


class MyLLM(MyKnowledge):
    __chat_history = ChatMessageHistory()
    print('__chat_history:', __chat_history)

    def get_chain(self, collection, model, max_length, temperature):
        retriever = None
        print('collection:', collection)
        if collection:
            retriever = self.get_retrievers(collection)
            print('retriever:', retriever)

        # 只保留3个记录
        print('len:', self.__chat_history.messages, '####:', len(self.__chat_history.messages))
        if len(self.__chat_history.messages) > 6:
            self.__chat_history.messages = self.__chat_history.messages[-6:]

        chat = ChatOpenAI(model=model, max_tokens=max_length, temperature=temperature)

        if retriever:
            question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
            print('question_answer_chain:', question_answer_chain)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            print('rag_chain:', rag_chain)
        else:
            rag_chain = normal_prompt | chat | streaming_parse
            print('rag_chain:', rag_chain)
        ''' 需要注意：output_messages_key，如果是无知识库的情况下是从AIMessageChunk的Content取，
            知识库是返回 AddableDict('answer') '''

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.__chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        print('conversational_rag_chain:', conversational_rag_chain)
        return conversational_rag_chain

    def invoke(self, question, collection, model="gpt-3.5-turbo", max_length=256, temperature=1):
        return self.get_chain(collection, model, max_length, temperature).invoke(
            {"input": question},
            {"configurable": {"session_id": "unused"}},
        )

    def stream(self, question, collection, model="gpt-3.5-turbo", max_length=256, temperature=1):
        return self.get_chain(collection, model, max_length, temperature).stream(
            {"input": question},
            {"configurable": {"session_id": "unused"}},
        )

    def clear_history(self) -> None:
        self.__chat_history.clear()

    def get_history_message(self):
        return self.__chat_history.messages


if __name__ == "__main__":
    pass
