"""
Enhanced Agentic Memory System with Multi-Doc QA
Combines conversational memory with external document retrieval
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional, Dict
from datetime import datetime
from pathlib import Path

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    WebBaseLoader, DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb

# =======================
# STATE DEFINITION
# =======================

class AgentState(TypedDict):
    """Enhanced state object with document support"""
    query: str
    context: Optional[str]
    context_source: Optional[str]  # 'short', 'long', or 'document'
    document_sources: Optional[List[Dict]]  # Track which docs were used
    alignment: Optional[bool]
    answer: Optional[str]
    summary: Optional[str]
    context_name: str
    use_documents: bool  # Flag to search documents


# =======================
# CONFIGURATION
# =======================

class Config:
    """Configuration for models and databases"""
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/phi-2"
    
    # Database paths
    SHORT_MEMORY_PATH = "./chroma_db_short"
    LONG_MEMORY_PATH = "./chroma_db_long"
    DOCUMENT_MEMORY_PATH = "./chroma_db_documents"  # NEW: Document store
    
    # Document ingestion
    DOCUMENT_LIBRARY_PATH = "./document_library"  # Where to store uploaded docs
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search parameters
    TOP_K_RESULTS = 3
    SIMILARITY_THRESHOLD = 0.7
    DOCUMENT_TOP_K = 5  # Retrieve more from documents


# =======================
# INITIALIZE MODELS
# =======================

class ModelManager:
    """Manages local HuggingFace models"""
    
    def __init__(self):
        print("🔧 Initializing models (this may take a moment on first run)...")
        
        # Initialize embeddings model
        print(f"📥 Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM
        print(f"📥 Loading LLM: {Config.LLM_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype="auto"
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  # Increased for document synthesis
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        print("✅ Models loaded successfully!\n")
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_llm(self):
        return self.llm


# =======================
# DOCUMENT PROCESSOR
# =======================

class DocumentProcessor:
    """Handles document ingestion and processing"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Create document library directory
        os.makedirs(Config.DOCUMENT_LIBRARY_PATH, exist_ok=True)
        print("📚 Document processor initialized")
    
    def load_document(self, file_path: str) -> List:
        """Load document based on file type"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            elif file_ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            documents = loader.load()
            print(f"   ✓ Loaded {len(documents)} pages from {Path(file_path).name}")
            return documents
        except Exception as e:
            print(f"   ✗ Error loading {file_path}: {str(e)}")
            return []
    
    def load_from_url(self, url: str) -> List:
        """Load content from web URL"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"   ✓ Loaded content from {url}")
            return documents
        except Exception as e:
            print(f"   ✗ Error loading URL {url}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List:
        """Load all supported documents from directory"""
        all_docs = []
        
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            all_docs.extend(pdf_loader.load())
        except Exception as e:
            print(f"   ⚠ PDF loading warning: {str(e)}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            all_docs.extend(txt_loader.load())
        except Exception as e:
            print(f"   ⚠ TXT loading warning: {str(e)}")
        
        print(f"   ✓ Loaded {len(all_docs)} documents from directory")
        return all_docs
    
    def chunk_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks")
        return chunks


# =======================
# ENHANCED DATABASE MANAGERS
# =======================

class EnhancedMemoryDatabase:
    """Manages three vector databases: short memory, long memory, and documents"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
        # Initialize short memory (summaries)
        self.short_memory = Chroma(
            collection_name="short_memory",
            embedding_function=embeddings,
            persist_directory=Config.SHORT_MEMORY_PATH
        )
        
        # Initialize long memory (full conversations)
        self.long_memory = Chroma(
            collection_name="long_memory",
            embedding_function=embeddings,
            persist_directory=Config.LONG_MEMORY_PATH
        )
        
        # Initialize document memory (external documents)
        self.document_memory = Chroma(
            collection_name="document_memory",
            embedding_function=embeddings,
            persist_directory=Config.DOCUMENT_MEMORY_PATH
        )
        
        print("💾 Enhanced memory databases initialized (3 databases)")
    
    def search_short_memory(self, query: str, k: int = Config.TOP_K_RESULTS):
        """Search short memory (summaries)"""
        results = self.short_memory.similarity_search_with_score(query, k=k)
        if results and results[0][1] < Config.SIMILARITY_THRESHOLD:
            return results[0][0].page_content, results[0][0].metadata
        return None, None
    
    def search_long_memory(self, query: str, k: int = Config.TOP_K_RESULTS):
        """Search long memory (full conversations)"""
        results = self.long_memory.similarity_search_with_score(query, k=k)
        if results and results[0][1] < Config.SIMILARITY_THRESHOLD:
            return results[0][0].page_content, results[0][0].metadata
        return None, None
    
    def search_documents(self, query: str, k: int = Config.DOCUMENT_TOP_K):
        """Search document memory - returns multiple relevant chunks"""
        results = self.document_memory.similarity_search_with_score(query, k=k)
        
        # Filter by threshold and format results
        relevant_docs = []
        for doc, score in results:
            if score < Config.SIMILARITY_THRESHOLD:
                relevant_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
        
        return relevant_docs if relevant_docs else None
    
    def save_to_short_memory(self, context_name: str, summary: str):
        """Save summary to short memory"""
        metadata = {
            "context_name": context_name,
            "timestamp": datetime.now().isoformat(),
            "type": "summary"
        }
        self.short_memory.add_texts([summary], metadatas=[metadata])
        print(f"💾 Saved to short memory: {context_name}")
    
    def save_to_long_memory(self, context_name: str, full_conversation: str):
        """Save full conversation to long memory"""
        metadata = {
            "context_name": context_name,
            "timestamp": datetime.now().isoformat(),
            "type": "full_conversation"
        }
        self.long_memory.add_texts([full_conversation], metadatas=[metadata])
        print(f"💾 Saved to long memory: {context_name}")
    
    # def add_documents(self, chunks: List):
    #     """Add document chunks to document memory"""
    #     texts = [chunk.page_content for chunk in chunks]
    #     metadatas = [chunk.metadata for chunk in chunks]
        
    #     self.document_memory.add_texts(texts, metadatas=metadatas)
    #     print(f"📚 Added {len(chunks)} chunks to document memory")

    def add_documents(self, docs, metadatas=None, batch_size=5000):
        """Add documents to ChromaDB in batches to avoid exceeding max_batch_size."""
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_metas = [doc.metadata for doc in batch_docs]
            self.document_memory.add_texts(batch_texts, metadatas=batch_metas)
    
    def get_document_stats(self):
        """Get statistics about stored documents"""
        # This is a simplified version - ChromaDB doesn't expose count easily
        return {
            "document_count": "Available",
            "status": "Active"
        }


# =======================
# ENHANCED AGENT NODES
# =======================

class EnhancedAgentNodes:
    """Enhanced agents with document retrieval support"""
    
    def __init__(self, llm, memory_db):
        self.llm = llm
        self.memory_db = memory_db
    
    def context_satisfier(self, state: AgentState) -> AgentState:
        """Agent 2: Enhanced Context Satisfier - Search memory AND documents"""
        print("\n🔍 Agent 2: Enhanced Context Satisfier")
        query = state['query']
        
        # First, search short memory
        print("   → Searching short memory...")
        context, metadata = self.memory_db.search_short_memory(query)
        
        if context:
            print(f"   ✓ Found in short memory: {metadata.get('context_name', 'unknown')}")
            state['context'] = context
            state['context_source'] = 'short'
            state['document_sources'] = None
            return state
        
        # If not found, search long memory
        print("   → No match in short memory, searching long memory...")
        context, metadata = self.memory_db.search_long_memory(query)
        
        if context:
            print(f"   ✓ Found in long memory: {metadata.get('context_name', 'unknown')}")
            state['context'] = context
            state['context_source'] = 'long'
            state['document_sources'] = None
            return state
        
        # NEW: Search documents if enabled
        if state.get('use_documents', True):
            print("   → Searching document library...")
            doc_results = self.memory_db.search_documents(query)
            
            if doc_results:
                print(f"   ✓ Found {len(doc_results)} relevant document chunks")
                
                # Combine multiple document chunks
                combined_context = "\n\n---\n\n".join([
                    f"[Source: {doc['metadata'].get('source', 'unknown')}]\n{doc['content']}"
                    for doc in doc_results
                ])
                
                state['context'] = combined_context
                state['context_source'] = 'document'
                state['document_sources'] = doc_results
                return state
        
        print("   ✗ No relevant context found anywhere")
        state['context'] = None
        state['context_source'] = None
        state['document_sources'] = None
        return state
    
    def context_alignment_checker(self, state: AgentState) -> AgentState:
        """Agent 1: Context Alignment Checker - Enhanced for documents"""
        print("\n🎯 Agent 1: Context Alignment Checker")
        
        if state['context'] is None:
            print("   → No context to check, skipping alignment")
            state['alignment'] = None
            return state
        
        # For documents, we're more lenient since they're curated content
        if state['context_source'] == 'document':
            print("   → Document source detected, using lenient alignment")
            # Still check but with simpler prompt
            prompt = f"""Does this document content relate to the question?
Reply only 'yes' or 'no'.

Question: {state['query']}

Document snippet: {state['context'][:300]}...

Answer:"""
        else:
            # Standard alignment check for memory
            prompt = f"""Question: Does the following context directly relate to the user's query?
Reply with only 'yes' or 'no'.

User Query: {state['query']}

Context: {state['context'][:500]}...

Answer (yes/no):"""
        
        response = self.llm.invoke(prompt)
        answer = response.lower().strip()
        
        is_aligned = 'yes' in answer
        
        print(f"   → Alignment check: {'✓ ALIGNED' if is_aligned else '✗ NOT ALIGNED'}")
        state['alignment'] = is_aligned
        
        if not is_aligned:
            state['context'] = None
        
        return state
    
    def answer_giver(self, state: AgentState) -> AgentState:
        """Agent 4: Enhanced Answer Giver - Handles document citations"""
        print("\n💬 Agent 4: Enhanced Answer Giver")
        
        if state['context'] and state['alignment']:
            # Different prompt based on context source
            if state['context_source'] == 'document':
                prompt = f"""Answer the question using ONLY the information from the provided documents.
Include which sources you used in your answer.

Documents:
{state['context']}

Question: {state['query']}

Answer with citations:"""
                print("   → Generating answer from DOCUMENTS")
            else:
                prompt = f"""Use the following context from our previous conversation to answer the question.

Context: {state['context']}

Question: {state['query']}

Answer:"""
                print("   → Generating answer from MEMORY")
        else:
            prompt = f"""Answer the following question to the best of your ability.

Question: {state['query']}

Answer:"""
            print("   → Generating answer from BASE KNOWLEDGE")
        
        response = self.llm.invoke(prompt)
        state['answer'] = response.strip()
        
        # Add citation info if from documents
        if state['context_source'] == 'document' and state['document_sources']:
            sources = set([doc['metadata'].get('source', 'unknown') 
                          for doc in state['document_sources']])
            citation = f"\n\n📚 Sources: {', '.join(sources)}"
            state['answer'] += citation
        
        print(f"   ✓ Answer generated ({len(response)} chars)")
        
        return state
    
    def context_summary_maker(self, state: AgentState) -> AgentState:
        """Agent 3: Context & Summary Maker - Enhanced with source tracking"""
        print("\n📝 Agent 3: Context & Summary Maker")
        
        # Create full conversation log with source info
        source_info = f"Context Source: {state['context_source'] or 'None'}"
        if state['document_sources']:
            source_info += f"\nDocuments Used: {len(state['document_sources'])} chunks"
        
        full_conversation = f"""Query: {state['query']}
Answer: {state['answer']}
Timestamp: {datetime.now().isoformat()}
{source_info}"""
        
        # Generate summary
        summary_prompt = f"""Summarize this conversation in 2-3 sentences:

{full_conversation}

Summary:"""
        
        summary = self.llm.invoke(summary_prompt).strip()
        state['summary'] = summary
        
        # Generate context name
        context_name = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state['context_name'] = context_name
        
        # Save to databases
        self.memory_db.save_to_short_memory(context_name, summary)
        self.memory_db.save_to_long_memory(context_name, full_conversation)
        
        print("   ✓ Memory updated successfully")
        
        return state


# =======================
# ROUTING LOGIC
# =======================

def should_check_alignment(state: AgentState) -> str:
    """Decide whether to check alignment or skip to answer"""
    if state['context'] is not None:
        return "check_alignment"
    else:
        return "generate_answer"

def alignment_result_router(state: AgentState) -> str:
    """Route based on alignment check result"""
    if state['alignment'] is True:
        return "generate_answer"
    elif state['context_source'] == 'short':
        return "retry_long_memory"
    else:
        return "generate_answer"


# =======================
# BUILD ENHANCED LANGGRAPH
# =======================

def build_enhanced_agent_graph(model_manager: ModelManager, memory_db: EnhancedMemoryDatabase):
    """Build the enhanced LangGraph workflow with document support"""
    
    agents = EnhancedAgentNodes(model_manager.get_llm(), memory_db)
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("context_satisfier", agents.context_satisfier)
    workflow.add_node("alignment_checker", agents.context_alignment_checker)
    workflow.add_node("answer_giver", agents.answer_giver)
    workflow.add_node("summary_maker", agents.context_summary_maker)
    
    # Set entry point
    workflow.set_entry_point("context_satisfier")
    
    # Add edges
    workflow.add_conditional_edges(
        "context_satisfier",
        should_check_alignment,
        {
            "check_alignment": "alignment_checker",
            "generate_answer": "answer_giver"
        }
    )
    
    workflow.add_conditional_edges(
        "alignment_checker",
        alignment_result_router,
        {
            "generate_answer": "answer_giver",
            "retry_long_memory": "context_satisfier"
        }
    )
    
    workflow.add_edge("answer_giver", "summary_maker")
    workflow.add_edge("summary_maker", END)
    
    return workflow.compile()


# =======================
# ENHANCED MAIN SYSTEM
# =======================

class EnhancedAgenticMemorySystem:
    """Enhanced system with Multi-Doc QA support"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Agentic Memory System with Multi-Doc QA...\n")
        self.model_manager = ModelManager()
        self.memory_db = EnhancedMemoryDatabase(self.model_manager.get_embeddings())
        self.doc_processor = DocumentProcessor(self.model_manager.get_embeddings())
        self.graph = build_enhanced_agent_graph(self.model_manager, self.memory_db)
        print("\n✅ Enhanced system ready!\n")
    
    def ingest_document(self, file_path: str):
        """Ingest a single document into the system"""
        print(f"\n📄 Ingesting document: {file_path}")
        docs = self.doc_processor.load_document(file_path)
        if docs:
            chunks = self.doc_processor.chunk_documents(docs)
            self.memory_db.add_documents(chunks)
            print("✅ Document ingested successfully!")
        else:
            print("❌ Failed to ingest document")
    
    def ingest_url(self, url: str):
        """Ingest content from a URL"""
        print(f"\n🌐 Ingesting content from: {url}")
        docs = self.doc_processor.load_from_url(url)
        if docs:
            chunks = self.doc_processor.chunk_documents(docs)
            self.memory_db.add_documents(chunks)
            print("✅ URL content ingested successfully!")
        else:
            print("❌ Failed to ingest URL content")
    
    def ingest_directory(self, directory_path: str):
        """Ingest all documents from a directory"""
        print(f"\n📁 Ingesting all documents from: {directory_path}")
        docs = self.doc_processor.load_directory(directory_path)
        if docs:
            chunks = self.doc_processor.chunk_documents(docs)
            self.memory_db.add_documents(chunks)
            print("✅ Directory documents ingested successfully!")
        else:
            print("❌ No documents found in directory")
    
    def query(self, user_query: str, use_documents: bool = True) -> dict:
        """Process a user query through the enhanced agent system"""
        print(f"\n{'='*60}")
        print(f"📨 User Query: {user_query}")
        print(f"{'='*60}")
        
        initial_state = {
            "query": user_query,
            "context": None,
            "context_source": None,
            "document_sources": None,
            "alignment": None,
            "answer": None,
            "summary": None,
            "context_name": "",
            "use_documents": use_documents
        }
        
        result = self.graph.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ FINAL ANSWER:")
        print(f"{'='*60}")
        print(result['answer'])
        print(f"{'='*60}\n")
        
        return result
    
    def get_stats(self):
        """Get system statistics"""
        return {
            "document_memory": self.memory_db.get_document_stats(),
            "status": "Active"
        }


# =======================
# USAGE EXAMPLE
# =======================

if __name__ == "__main__":
    # Initialize enhanced system
    system = EnhancedAgenticMemorySystem()
    
    # Example 1: Ingest some documents
    print("\n" + "="*60)
    print("📚 STEP 1: Ingest Documents")
    print("="*60)
    
    # You can ingest from:
    # system.ingest_document("path/to/document.pdf")
    # system.ingest_url("https://example.com/article")
    # system.ingest_directory("./my_documents")
    
    print("\n💡 To ingest documents, uncomment the lines above and provide paths")
    print("   For this demo, we'll proceed with queries only\n")
    
    # Example 2: Query the system
    print("\n" + "="*60)
    print("🎯 STEP 2: Running Example Queries")
    print("="*60 + "\n")
    
    queries = [
        "What is machine learning?",
        "How does supervised learning work?",
        "What did we discuss about machine learning?"
    ]
    
    for query in queries:
        result = system.query(query)
        input("\n⏸️  Press Enter to continue to next query...\n")
    
    print("\n✅ Demo complete!")
    print("\n💡 Tips:")
    print("   - Use system.ingest_document() to add PDFs, TXT, DOCX files")
    print("   - Use system.ingest_url() to add web content")
    print("   - Use system.ingest_directory() to batch-add documents")
    print("   - The system will search documents AND maintain conversation memory!")