import json
from abc import ABC
from typing import List, Dict, Any, Optional

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra


class WriteToAstra(Chain, ABC):
  """
  A chain to write to astra vector db.
  """

  astra_db_token_path: str
  """The astra db token path."""
  astra_db_secure_bundle_path: str
  """The astra db secure bundle path."""
  keyspace_name: str
  """The keyspace name."""
  table_name: str
  """The table name."""

  embedding_model: str
  """Model to use to create embeddings."""
  #openai_api_base: str
  """The base URL for the OpenAI API."""
  #openai_api_type: str = "azure"
  """The type of OpenAI."""
  #openai_api_key: str = "aaaa"
  """The key for OpenAI API."""

  class Config:
    """Configuration for this pydantic object."""

  @property
  def input_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return ["docs"]

  @property
  def output_keys(self) -> List[str]:
    """Will always return text key.

    :meta private:
    """
    return []

  @property
  def _chain_type(self) -> str:
    return "write_to_astra"

  def _call(
      self,
      inputs: Dict[str, Any],
      run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, List[Any]]:
    try:
      cloud_config = {
        'secure_connect_bundle': self.astra_db_secure_bundle_path
      }

      with open(self.astra_db_token_path) as f:
        secrets = json.load(f)

      CLIENT_ID = secrets["clientId"]
      CLIENT_SECRET = secrets["secret"]

      auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
      cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
      session = cluster.connect()
    except Exception as e:
      print(e)
      raise Exception("Error connecting to Astra DB") from e

    embeddings_model = OpenAIEmbeddings(
      model=self.embedding_model,
      #openai_api_base=self.openai_api_base,
      #openai_api_type=self.openai_api_type,
      #openai_api_key=self.openai_api_key,
    )

    try:
      Cassandra.from_documents(
        documents=inputs["docs"],
        embedding=embeddings_model,
        session=session,
        keyspace=self.keyspace_name,
        table_name=self.table_name,
      )
    except Exception as e:
      raise Exception("Error writing to Astra DB") from e

    return {}
