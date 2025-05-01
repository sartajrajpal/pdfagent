import os
import sys
import shutil
from typing import List, Dict, Any
from dotenv import load_dotenv
import arxiv
import requests
from Bio import Entrez
from cachetools import cached, TTLCache
from diskcache import Cache
import hashlib
from pathlib import Path
import logging
import subprocess 
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document

load_dotenv()

Entrez.email = os.getenv('PUBMED_EMAIL', 'sartaj.rajpal@yale.edu')

class ResearchAgent:
    def __init__(self):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            raise ValueError("OpenAI API key not found")
        
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.cache_dir = Path('cache/pdfs')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache = Cache(str(self.cache_dir))
        self.output_dir = Path('data/converted')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("ResearchAgent initialized successfully")

    def clear_output_directory(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("Cleared and recreated the output directory")

    def _get_cache_key(self, url: str):
        return hashlib.md5(url.encode()).hexdigest()

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def search_pubmed(self, query: str, max_results: int = 5):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        if not record['IdList']:
            print(f"No results found in PubMed for query: {query}")
            return []
            
        handle = Entrez.efetch(db="pubmed", id=record['IdList'], rettype="medline", retmode="text")
        records = Entrez.parse(handle)
        
        results = []
        for record in records:
            results.append({
                'title': record.get('TI', ''),
                'authors': record.get('AU', []),
                'abstract': record.get('AB', ''),
                'pmid': record.get('PMID', ''),
                'doi': record.get('AID', [''])[0].split(' [doi]')[0] if 'AID' in record else ''
            })
        print(f"Found {len(results)} papers in PubMed")
        return results

    def search_arxiv(self, query: str, max_results: int = 5):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            results.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published,
                'source': 'arxiv'
            })
        print(f"Found {len(results)} papers in arXiv")
        return results

    def process_pdf(self, pdf_url: str):
        cache_key = self._get_cache_key(pdf_url)
        cached_text = self.disk_cache.get(cache_key)
        if cached_text:
            print(f"Retrieved PDF from cache: {pdf_url}")
            return cached_text
        
        print(f"Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        pdf_path = self.output_dir / f"{os.path.basename(pdf_url)}"
        with open(pdf_path, 'wb') as temp_file:
            temp_file.write(response.content)
        
        print(f"Processing PDF with Marker: {pdf_url}")
        markdown_text = self.convert_pdf_to_markdown(pdf_path)
        
        self.disk_cache.set(cache_key, markdown_text)
        print(f"Cached PDF: {pdf_url}")
        return markdown_text
       

    def convert_pdf_to_markdown(self, pdf_path: str:
        command = f"marker_single \"{pdf_path}\" --output_dir \"{self.output_dir}\""
        subprocess.run(command, shell=True, check=True)
        
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_file_path = self.output_dir / base_name / f"{base_name}.md"
        
        if markdown_file_path.exists():
            with open(markdown_file_path, 'r') as f:
                return f.read()
        else:
            print(f"Markdown file not found: {markdown_file_path}")
            return ""
        

    def create_vector_store(self, documents: List[Document]):
        if not documents:
            print("No documents to create vector store")
            return
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"Created vector store with {len(documents)} documents")
        

    def get_relevant_documents(self, query: str, k: int = 3):
        if not self.vector_store:
            print("No vector store available")
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        print(f"Retrieved {len(docs)} relevant documents")
        return docs


    def process_query(self, query: str):
        print(f"Processing query: {query}")
        
        self.clear_output_directory()
        
        arxiv_papers = self.search_arxiv(query)
        
        if arxiv_papers:
            pdf_url = arxiv_papers[0]['pdf_url']
            print(f"Processing PDF from arXiv: {pdf_url}")
            pdf_path = self.output_dir / f"{os.path.basename(pdf_url)}"
            
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            markdown_text = self.convert_pdf_to_markdown(pdf_path)
            if markdown_text:
                output = f"Question: {query}\n\nMarkdown Content:\n{markdown_text}"
                print("Output generated successfully")
                return output
            else:
                print("Failed to convert PDF to markdown.")
                return "Failed to convert the PDF to markdown."
        
        print("No arXiv papers found, checking PubMed...")
        pubmed_papers = self.search_pubmed(query)
        
        if pubmed_papers:
            pdf_url = pubmed_papers[0]['doi'] 
            print(f"Processing PDF from PubMed: {pdf_url}")
            pdf_path = self.output_dir / f"{pubmed_papers[0]['pmid']}.pdf" 
            
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            markdown_text = self.convert_pdf_to_markdown(pdf_path)
            if markdown_text:
                output = f"Question: {query}\n\nMarkdown Content:\n{markdown_text}"
                print("Output generated successfully")
                return output
            else:
                print("Failed to convert PDF to markdown.")
                return "Failed to convert the PDF to markdown."
        
        print("No relevant research papers found for this query.")
        return "No relevant research papers found for this query."
    
def main():
    agent = ResearchAgent()
    # QUESTION
    query = "What is the likelihood that a patient with high troponin levels might develop coronary artery disease?"
    response = agent.process_query(query)
    
    print(f"Question: {query}\nResponse: {response}")
    

if __name__ == "__main__":
    main()