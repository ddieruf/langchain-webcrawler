import argparse
import json
import sys

from dotenv import load_dotenv

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import SequentialChain, RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Cassandra

from chains.crawler import Crawler
from chains.writeToAstra import WriteToAstra


def crawl_site():
  crawler_chain = Crawler(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=400,
    chunk_overlap=10,
    encoding_name="cl100k_base",
    embedding_model="gpt-3.5-turbo",
    spacy_model_name="en_core_web_sm",
    allowed_languages=["en"],
    output_key="docs",
  )

  write_to_astra_chain = WriteToAstra(
    astra_db_token_path="langstream_webcrawler-token.json",
    astra_db_secure_bundle_path="secure-connect-langstream-webcrawler.zip",
    keyspace_name="langstream_webcrawler",
    table_name="docs",
    embedding_model="text-embedding-ada-002",

    #  From .env file
    #openai_api_type
    #openai_api_base
    #openai_api_key
  )

  overall_chain = SequentialChain(
    chains=[crawler_chain, write_to_astra_chain],
    input_variables=["urls"],
    output_variables=["docs"],
    verbose=False,
  )

  # drop table langstream_webcrawler.docs ;
  overall_chain.run({"urls": ["https://docs.langstream.ai"]}, callbacks=[StdOutCallbackHandler()])

  return


def chat(question: str):
  if not question:
    return "Please ask a question"

  try:
    cloud_config = {
      'secure_connect_bundle': "secure-connect-langstream-webcrawler.zip"
    }

    with open("langstream_webcrawler-token.json") as f:
      secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
  except Exception as e:
    print(e)
    raise Exception("Error connecting to Astra DB") from e

  embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",

    #  From .env file
    #openai_api_type
    #openai_api_base
    #openai_api_key
  )

  # https://python.langchain.com/docs/integrations/vectorstores/cassandra
  cassandra_vector_store = Cassandra(
    embedding=embeddings_model,
    session=cluster.connect(),
    keyspace="langstream_webcrawler",
    table_name="docs",
  )

  template = """Use the following pieces of context to answer the question at the end. 
  If you don't know the answer, just say that you don't know, don't try to make up an answer. 
  Use three sentences maximum and keep the answer as concise as possible. 
  Always say "thanks for asking!" at the end of the answer. 
  {context}

  Question: {question}

  Helpful Answer:"""

  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

  #  https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.azure_openai.AzureChatOpenAI.html
  llm = AzureChatOpenAI(
    model_name="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    temperature=0,
    verbose=True,

    #  From .env file
    #openai_api_version
    #openai_api_type
    #openai_api_base
    #openai_api_key
  )

  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=cassandra_vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 2, 'score_threshold': 0.909}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    verbose=True,
  )

  return qa_chain({"query": question}, callbacks=[StdOutCallbackHandler()])


def parse_args(args):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--action',
    help='What chain to run',
    choices=['crawl-site', 'chat']
  )

  return parser.parse_args(args)


def main(params=None):
  if params is None:
    params = sys.argv[1:]

  args = parse_args(params)

  load_dotenv()

  if args.action is None or args.action == "":
    print("Please specify an action or 'crawl-site' or 'chat'")
    return 0

  if args.action == 'crawl-site':
    answer = input("Did you clear the table? (y/n): ")

    if answer.lower() != "y":
      print("Please clear the table before running the crawler")
      print('drop table langstream_webcrawler.docs ;')
      return 0

    crawl_site()

  if args.action == 'chat':
    while True:
      question = input("\nAsk a question (or type 'quit' to exit): ")

      if question.lower() == "quit":
        break

      answer = chat(question)
      print(json.dumps(answer, indent=2))

  return 0


if __name__ == '__main__':
  sys.exit(main())
