from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import prompt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import textwrap
import gradio
from langchain.vectorstores import Chroma
import os
from torch import cuda

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OPENAI_API_KEY"] = "dummy-key"


# TODO: Callbacks support token-wise streaming
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

temperature = 0.01 # Use a value between 0 and 2. Lower = factual, higher = creative
n_gpu_layers = 43  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = OpenAI(
       #openai_api_base='http://localhost:1234/v1',
       openai_api_base='http://192.168.2.47:1234/v1',
       openai_api_key='dummy-key',
       max_tokens=2048,
       temperature = 0.01
       )

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

## Follow the default prompt style from the OpenOrca-Platypus2 huggingface model card.

def get_prompt():
  return """Use the following Context information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
### Instruction:
Context: {context}
User Question: {question}
###
Response:
"""

# def get_prompt():
#   return """Используй Context для ответа на User Question. Если не знаешь, что ответить - напиши, что не можешь ответить. Отвечай на русском языке.
# ### Instruction:
# Context: {context}
# User Question: {question}
# ###
# Response:
# """

# def get_prompt():
#    return """### Instruction:
# Context: {context}
# User Question: {question}
# ### Response:
# """

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    if not llm_response:
      return "Please enter a question"
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    response = llm_response['result']
    response = response.split("### Response")[0]
    return response

def startChat():

   embedding_directory = "./content/chroma_db"
   #embedding_model=HuggingFaceBgeEmbeddings(model_name='BAAI/bge-base-en', model_kwargs={'device':device})
   embedding_model=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_kwargs={'device':device})
   embedding_db = Chroma(persist_directory=embedding_directory, embedding_function=embedding_model)

   prompt_template = get_prompt()

   llama_prompt = PromptTemplate(
       template=prompt_template, input_variables=["context", "question"]
   )

   chain_type_kwargs = {"prompt": llama_prompt}


   retriever = embedding_db.as_retriever(search_type="mmr", search_kwargs={'k': 5})

   # create the chain to answer questions
   qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       chain_type_kwargs=chain_type_kwargs,
                                       return_source_documents=True)
    #  query = '''Ответь на 6 вопросов:
    #  1. Каким трем критериям должен соответствовать атрибут, чтобы считаться архитектурным свойством?
    #  2. В чем разница между явным и неявным свойством? Приведите пример каждого из них.
    #  3. Приведите пример эксплуатационных свойств.
    #  4. Приведите пример структурных свойств.
    #  5. Приведите пример сквозных свойств.
    #  6. Какое архитектурное свойство важнее: доступность или производительность?
    #  Отвечай на русском языке.
    #  '''

  #  with open('/tmp/my1_qwest.txt', 'r') as f:
  #     lines = f.readlines()

  #  for query in lines:
  #   # query = 'Назовите четыре главные особенности, определяющие архитектуру программного обеспечения.'
  #   response = qa_chain( query + ' Отвечай на русском языке.' )
  #   print( query)
  #   print( response["result"])




   def runChain(query, history):
     return process_llm_response(qa_chain(query))

   app = gradio.ChatInterface(runChain)
   app.queue()
   app.launch(share=False, debug=True)


startChat()
