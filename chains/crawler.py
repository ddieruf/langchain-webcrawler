from abc import ABC
from typing import Any, Dict, Optional, List

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacy_download import load_spacy
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


class Crawler(Chain, ABC):
  """
  A web crawler chain.
  """

  spacy_model_name: str = "en_core_web_sm"
  """The spacy language model to use."""
  allowed_languages: list[str] = ["en"]
  """Allowed languages discovered by Spacy language detector."""
  separators: list[str] = ["\n\n", "\n", " ", ""]
  """The seperator(s) to use for splitting the text. Do not use regex."""
  chunk_size: int
  """The chunk size to use while splitting the text."""
  chunk_overlap: int
  """The chunk overlap to use while splitting the text."""
  encoding_name: str = "cl100k_base"
  """Tokenizer to use to create embeddings."""
  embedding_model: str = "gpt-3.5-turbo"
  """Model to use to create embeddings."""

  output_key: str = "docs"  #: :meta private:

  class Config:
    """Configuration for this pydantic object."""

  @property
  def input_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return ["urls"]

  @property
  def output_keys(self) -> List[str]:
    """Will always return text key.

    :meta private:
    """
    return [self.output_key]

  @property
  def _chain_type(self) -> str:
    return "web_crawler"

  @Language.factory("language_detector")
  def get_lang_detector(nlp, name):
    return LanguageDetector()

  def _call(
      self,
      inputs: Dict[str, Any],
      run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, List[Document]]:
    """
    Call the crawler chain.

    :param inputs: The starting URL(s) to begin crawling.
    :param run_manager: Call back manager.
    :return: Embedding documents.
    """

    if run_manager:
      run_manager.on_text("Beginning to crawl site...")

    try:
      # Crawl Site
      loader = AsyncHtmlLoader(inputs["urls"])
      html = loader.load()
    except Exception as e:
      raise Exception("Error crawling site") from e

    if run_manager:
      run_manager.on_text("Rendering HTML...")

    try:
      # Render HTML
      bs_transformer = BeautifulSoupTransformer()
      docs_transformed = bs_transformer.transform_documents(html)
    except Exception as e:
      raise Exception("Error rendering HTML") from e

    nlp = load_spacy(self.spacy_model_name)
    nlp.add_pipe('language_detector', last=True)

    if run_manager:
      run_manager.on_text("Ensuring document language is supported...")

    for html_doc in docs_transformed:
      try:
        # Normalize Text
        html_doc.page_content = html_doc.page_content.lower()  # lower
        html_doc.page_content = html_doc.page_content.lstrip(' ').rstrip(' ')  # trim

        # Detect Language
        doc = nlp(html_doc.page_content)
      except Exception as e:
        raise Exception("Error detecting language") from e

      if doc._.language["language"] not in self.allowed_languages:  # doc._.language_score >= 0.8
        raise Exception("Language '{0}' not supported".format(doc._.language["language"]))

    if run_manager:
      run_manager.on_text("Splitting text...")

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
      separators=self.separators,
      chunk_size=self.chunk_size,
      chunk_overlap=self.chunk_overlap,
      length_function=len,
      is_separator_regex=False,
    )

    try:
      splitter = text_splitter.from_tiktoken_encoder(encoding_name=self.encoding_name, model_name=self.embedding_model)
      docs = splitter.split_documents(docs_transformed)

    except Exception as e:
      raise Exception("Error splitting text") from e

    if run_manager:
      run_manager.on_text("Successfully split text for crawled sites")

    return {self.output_key: docs}
