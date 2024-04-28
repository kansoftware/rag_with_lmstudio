import langchain.embeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import cuda
import sys

path = "./content"
embedding_directory = "./content/chroma_db"

embedding_db = None;

import os

def remove_first_line(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # read all the lines and skip the first one
    # with open(file, 'w') as f:
    #     for line in lines:
    #         f.write(line)
    return lines

def sorted_walk(top):
    for root, dirs, files in os.walk(top):
        yield (root, sorted(dirs), sorted(files))

def process_directory(path):
    lines = []
    for root, dirs, files in sorted_walk(path):
        for file in files:
            if file.endswith(".txt"):  # only process text files
                print( file)
                lines = lines + remove_first_line(os.path.join(root, file))
    return ''.join(lines)


def embed():
    chunk_size=256
    chunk_overlap=32
    print("\nCalculating Embeddings\n")

    # Load the text from the path
    # text_loader_kwargs = {'autodetect_encoding': True}
    # loader=DirectoryLoader(path,
    #                     glob="./*.txt",
    #                     loader_cls=TextLoader,  loader_kwargs=text_loader_kwargs)

    # documents=loader.load()

    document_txt = process_directory("./content/")
    with open("/tmp/my.txt", 'w') as f:
        for line in document_txt:
            f.write(line)


    loaders = [
        # PyPDFLoader('/home/kan/python/AutogenLangchainPDFchat/chat_docs4.pdf'),
        TextLoader("/tmp/my1.txt")
    ]
    
    docs = []
    for file in loaders:
        docs.extend(file.load())
    
    #split text to chunks
    #text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    

    # Split the data into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    # chunks = text_splitter.split_text(document_txt)

    # Load the huggingface embedding model
    # model_name = "BAAI/bge-base-en"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers/all-MiniLM-L6-v2
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embedding_model = langchain.embeddings.HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    embedding_db = Chroma.from_documents(chunks, embedding_model, persist_directory=embedding_directory)


    # loaders = [PyPDFLoader('/home/kan/python/AutogenLangchainPDFchat/chat_docs4.pdf')]
    # docs = []
    # for file in loaders:
    #     docs.extend(file.load())
    
    # #split text to chunks
    # #text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    # docs = text_splitter.split_documents(docs)
    # embedding_db.add_documents(docs)

    print("Embeddings completed")

embed()
