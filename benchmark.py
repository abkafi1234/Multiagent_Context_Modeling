"""
Agent Benchmarking System
Supports LongBench v2 and custom benchmarks for the Agentic Memory System
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
from enum import Enum

# For metrics
from collections import defaultdict
import re


# =======================
# BENCHMARK DATA STRUCTURES
# =======================

class BenchmarkType(Enum):
    """Types of benchmarks"""
    LONGBENCH_V2 = "longbench_v2"
    SINGLE_DOC_QA = "single_doc_qa"
    MULTI_DOC_QA = "multi_doc_qa"
    CONVERSATIONAL = "conversational"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CUSTOM = "custom"


@dataclass
class BenchmarkQuestion:
    """Single benchmark question"""
    id: str
    question: str
    context: Optional[str]
    ground_truth: str
    category: str
    difficulty: str
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result for a single question"""
    question_id: str
    question: str
    predicted_answer: str
    ground_truth: str
    category: str
    
    # Metrics
    exact_match: float
    f1_score: float
    semantic_similarity: float
    
    # Performance
    response_time: float
    context_source: str
    tokens_used: int
    
    # Memory usage
    memory_retrieved: bool
    document_retrieved: bool
    
    timestamp: str


@dataclass
class BenchmarkReport:
    """Complete benchmark report"""
    benchmark_name: str
    benchmark_type: str
    total_questions: int
    
    # Overall metrics
    avg_exact_match: float
    avg_f1_score: float
    avg_semantic_similarity: float
    avg_response_time: float
    
    # Breakdown by category
    category_results: Dict[str, Dict[str, float]]
    
    # Detailed results
    individual_results: List[BenchmarkResult]
    
    # System info
    system_config: Dict[str, Any]
    timestamp: str


# =======================
# METRIC CALCULATORS
# =======================

class MetricCalculator:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """Calculate exact match score"""
        pred_norm = MetricCalculator.normalize_text(prediction)
        truth_norm = MetricCalculator.normalize_text(ground_truth)
        return 1.0 if pred_norm == truth_norm else 0.0
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Calculate F1 score (token-level)"""
        pred_tokens = set(MetricCalculator.normalize_text(prediction).split())
        truth_tokens = set(MetricCalculator.normalize_text(ground_truth).split())
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        common_tokens = pred_tokens.intersection(truth_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def semantic_similarity_simple(prediction: str, ground_truth: str) -> float:
        """
        Simple semantic similarity based on word overlap
        For more accurate results, use embeddings-based similarity
        """
        pred_tokens = set(MetricCalculator.normalize_text(prediction).split())
        truth_tokens = set(MetricCalculator.normalize_text(ground_truth).split())
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        intersection = len(pred_tokens.intersection(truth_tokens))
        union = len(pred_tokens.union(truth_tokens))
        
        return intersection / union if union > 0 else 0.0


# =======================
# LONGBENCH V2 LOADER
# =======================

class LongBenchV2Loader:
    """Load and parse LongBench v2 datasets"""
    
    TASKS = [
        "single_doc_qa",
        "multi_doc_qa", 
        "summarization",
        "few_shot_learning",
        "synthetic",
        "code_completion"
    ]
    
    @staticmethod
    def load_from_file(file_path: str) -> List[BenchmarkQuestion]:
        """
        Load LongBench v2 format JSON file
        
        Expected format:
        [
            {
                "id": "question_1",
                "context": "long context text...",
                "question": "What is...?",
                "answer": "The answer is...",
                "category": "single_doc_qa",
                "length": 10000,
                "difficulty": "hard"
            },
            ...
        ]
        """
        questions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            question = BenchmarkQuestion(
                id=item.get('id', f"q_{len(questions)}"),
                question=item['question'],
                context=item.get('context', ''),
                ground_truth=item['answer'],
                category=item.get('category', 'unknown'),
                difficulty=item.get('difficulty', 'medium'),
                metadata={
                    'length': item.get('length', 0),
                    'source': item.get('source', 'longbench_v2'),
                    'extra': item.get('extra', {})
                }
            )
            questions.append(question)
        
        print(f"✅ Loaded {len(questions)} questions from {file_path}")
        return questions
    
    @staticmethod
    def load_from_huggingface(dataset_name: str = "THUDM/LongBench", split: str = "test") -> List[BenchmarkQuestion]:
        """Load LongBench v2 from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            
            print(f"📥 Loading {dataset_name} from HuggingFace...")
            
            # Try loading with different configurations
            try:
                dataset = load_dataset(dataset_name, split=split)
            except:
                # Try without split
                dataset = load_dataset(dataset_name)
                dataset = dataset[split] if split in dataset else dataset['test']
            
            print(f"   Dataset loaded, parsing {len(dataset)} items...")
            
            questions = []
            for i, item in enumerate(dataset):
                # Debug: print first item structure
                if i == 0:
                    print(f"   Sample item keys: {item.keys()}")
                
                # Flexible key mapping
                question_text = (
                    item.get('input') or 
                    item.get('question') or 
                    item.get('query') or 
                    ""
                )
                
                answer_text = (
                    item.get('answers') or 
                    item.get('answer') or 
                    item.get('output') or 
                    ""
                )
                
                context_text = (
                    item.get('context') or 
                    item.get('passages') or 
                    item.get('document') or 
                    ""
                )
                
                if not question_text or not answer_text:
                    print(f"   ⚠️  Skipping item {i}: missing question or answer")
                    continue
                
                question = BenchmarkQuestion(
                    id=item.get('id', f"hf_q_{i}"),
                    question=question_text,
                    context=context_text,
                    ground_truth=answer_text,
                    category=item.get('task', item.get('category', 'unknown')),
                    difficulty=item.get('difficulty', 'medium'),
                    metadata={
                        'length': len(context_text),
                        'source': 'huggingface',
                        'dataset': dataset_name
                    }
                )
                questions.append(question)
            
            print(f"✅ Successfully parsed {len(questions)} questions")
            return questions
            
        except ImportError:
            print("❌ Please install datasets: pip install datasets")
            return []
        except Exception as e:
            print(f"❌ Error loading from HuggingFace: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    @staticmethod
    def create_sample_benchmark() -> List[BenchmarkQuestion]:
        """Create a sample benchmark for testing"""
        return [
            BenchmarkQuestion(
                id="sample_1",
                question="What is the capital of France?",
                context="France is a country in Western Europe. Its capital and largest city is Paris.",
                ground_truth="Paris",
                category="single_doc_qa",
                difficulty="easy",
                metadata={"length": 100}
            ),
            BenchmarkQuestion(
                id="sample_2",
                question="What are the main components of a transformer architecture?",
                context="The transformer architecture consists of an encoder and decoder, each made up of multiple layers. Key components include self-attention mechanisms, feedforward networks, and positional encodings.",
                ground_truth="Self-attention mechanisms, feedforward networks, and positional encodings",
                category="single_doc_qa",
                difficulty="medium",
                metadata={"length": 200}
            ),
            BenchmarkQuestion(
                id="sample_3",
                question="Compare the benefits of supervised and unsupervised learning approaches.",
                context="Supervised learning uses labeled data to train models for prediction tasks. Unsupervised learning finds patterns in unlabeled data. Deep learning combines both approaches.",
                ground_truth="Supervised learning is good for prediction with labeled data, while unsupervised learning discovers patterns without labels",
                category="reasoning",
                difficulty="hard",
                metadata={"length": 150}
            )
        ]


# =======================
# BENCHMARK RUNNER
# =======================

class BenchmarkRunner:
    """Run benchmarks on the Agentic Memory System"""
    
    def __init__(self, agent_system):
        """
        Initialize benchmark runner
        
        Args:
            agent_system: Instance of EnhancedAgenticMemorySystem
        """
        self.agent_system = agent_system
        self.metric_calculator = MetricCalculator()
        self.results = []
    
    def run_benchmark(
        self,
        questions: List[BenchmarkQuestion],
        benchmark_name: str,
        benchmark_type: BenchmarkType = BenchmarkType.CUSTOM,
        ingest_contexts: bool = True,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Run a complete benchmark
        
        Args:
            questions: List of benchmark questions
            benchmark_name: Name of the benchmark
            benchmark_type: Type of benchmark
            ingest_contexts: Whether to ingest contexts as documents
            verbose: Print progress
        """
        print(f"\n{'='*70}")
        print(f"🏁 Starting Benchmark: {benchmark_name}")
        print(f"{'='*70}")
        print(f"📊 Total Questions: {len(questions)}")
        print(f"🏷️  Benchmark Type: {benchmark_type.value}")
        print(f"{'='*70}\n")
        
        # Step 1: Ingest contexts if needed
        if ingest_contexts:
            print("📚 Ingesting contexts into document memory...")
            self._ingest_contexts(questions)
            print("✅ Contexts ingested\n")
        
        # Step 2: Run each question
        results = []
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[{i}/{len(questions)}] Processing: {question.id}")
                print(f"   Category: {question.category} | Difficulty: {question.difficulty}")
            
            result = self._run_single_question(question, verbose)
            results.append(result)
            
            if verbose:
                print(f"   ✓ Exact Match: {result.exact_match:.2f}")
                print(f"   ✓ F1 Score: {result.f1_score:.2f}")
                print(f"   ✓ Time: {result.response_time:.2f}s")
        
        # Step 3: Calculate aggregate metrics
        print(f"\n{'='*70}")
        print("📈 Calculating aggregate metrics...")
        print(f"{'='*70}\n")
        
        report = self._generate_report(
            benchmark_name=benchmark_name,
            benchmark_type=benchmark_type.value,
            results=results
        )
        
        # Step 4: Display summary
        self._display_summary(report)
        
        return report
    
    def _ingest_contexts(self, questions: List[BenchmarkQuestion]):
        """Ingest all contexts as documents"""
        for question in questions:
            if question.context:
                # Create a temporary file with context
                context_id = f"benchmark_context_{question.id}"
                
                # Add to document memory with metadata
                from langchain_core.documents import Document
                doc = Document(
                    page_content=question.context,
                    metadata={
                        "source": context_id,
                        "question_id": question.id,
                        "category": question.category,
                        "benchmark": True
                    }
                )
                
                # Chunk and add to memory
                chunks = self.agent_system.doc_processor.text_splitter.split_documents([doc])
                self.agent_system.memory_db.add_documents(chunks)
    
    def _run_single_question(
        self,
        question: BenchmarkQuestion,
        verbose: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark question"""
        
        # Time the query
        start_time = time.time()
        
        try:
            # Query the system
            response = self.agent_system.query(question.question, use_documents=True)
            
            response_time = time.time() - start_time
            predicted_answer = response['answer']
            
            # Remove citations from answer for fair comparison
            if "📚 Sources:" in predicted_answer:
                predicted_answer = predicted_answer.split("📚 Sources:")[0].strip()
            
            # Calculate metrics
            exact_match = self.metric_calculator.exact_match(
                predicted_answer,
                question.ground_truth
            )
            
            f1_score = self.metric_calculator.f1_score(
                predicted_answer,
                question.ground_truth
            )
            
            semantic_sim = self.metric_calculator.semantic_similarity_simple(
                predicted_answer,
                question.ground_truth
            )
            
            # Create result
            result = BenchmarkResult(
                question_id=question.id,
                question=question.question,
                predicted_answer=predicted_answer,
                ground_truth=question.ground_truth,
                category=question.category,
                exact_match=exact_match,
                f1_score=f1_score,
                semantic_similarity=semantic_sim,
                response_time=response_time,
                context_source=response.get('context_source', 'unknown'),
                tokens_used=len(predicted_answer.split()),  # Approximate
                memory_retrieved=response.get('context_source') in ['short', 'long'],
                document_retrieved=response.get('context_source') == 'document',
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error processing question {question.id}: {str(e)}")
            
            # Return a failed result
            return BenchmarkResult(
                question_id=question.id,
                question=question.question,
                predicted_answer="[ERROR]",
                ground_truth=question.ground_truth,
                category=question.category,
                exact_match=0.0,
                f1_score=0.0,
                semantic_similarity=0.0,
                response_time=time.time() - start_time,
                context_source="error",
                tokens_used=0,
                memory_retrieved=False,
                document_retrieved=False,
                timestamp=datetime.now().isoformat()
            )
    
    def _generate_report(
        self,
        benchmark_name: str,
        benchmark_type: str,
        results: List[BenchmarkResult]
    ) -> BenchmarkReport:
        """Generate comprehensive benchmark report"""
        
        # Calculate overall metrics
        avg_exact_match = statistics.mean([r.exact_match for r in results])
        avg_f1_score = statistics.mean([r.f1_score for r in results])
        avg_semantic_sim = statistics.mean([r.semantic_similarity for r in results])
        avg_response_time = statistics.mean([r.response_time for r in results])
        
        # Calculate per-category metrics
        category_results = defaultdict(lambda: {
            'count': 0,
            'exact_match': [],
            'f1_score': [],
            'semantic_similarity': [],
            'response_time': []
        })
        
        for result in results:
            cat = result.category
            category_results[cat]['count'] += 1
            category_results[cat]['exact_match'].append(result.exact_match)
            category_results[cat]['f1_score'].append(result.f1_score)
            category_results[cat]['semantic_similarity'].append(result.semantic_similarity)
            category_results[cat]['response_time'].append(result.response_time)
        
        # Average per category
        category_avg = {}
        for cat, metrics in category_results.items():
            category_avg[cat] = {
                'count': metrics['count'],
                'avg_exact_match': statistics.mean(metrics['exact_match']),
                'avg_f1_score': statistics.mean(metrics['f1_score']),
                'avg_semantic_similarity': statistics.mean(metrics['semantic_similarity']),
                'avg_response_time': statistics.mean(metrics['response_time'])
            }
        
        # System config
        system_config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'microsoft/phi-2',
            'chunk_size': 1000,
            'top_k': 5
        }
        
        report = BenchmarkReport(
            benchmark_name=benchmark_name,
            benchmark_type=benchmark_type,
            total_questions=len(results),
            avg_exact_match=avg_exact_match,
            avg_f1_score=avg_f1_score,
            avg_semantic_similarity=avg_semantic_sim,
            avg_response_time=avg_response_time,
            category_results=category_avg,
            individual_results=results,
            system_config=system_config,
            timestamp=datetime.now().isoformat()
        )
        
        return report
    
    def _display_summary(self, report: BenchmarkReport):
        """Display benchmark summary"""
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK RESULTS: {report.benchmark_name}")
        print(f"{'='*70}\n")
        
        print(f"📈 Overall Metrics:")
        print(f"   Total Questions: {report.total_questions}")
        print(f"   Avg Exact Match: {report.avg_exact_match:.4f} ({report.avg_exact_match*100:.2f}%)")
        print(f"   Avg F1 Score: {report.avg_f1_score:.4f}")
        print(f"   Avg Semantic Similarity: {report.avg_semantic_similarity:.4f}")
        print(f"   Avg Response Time: {report.avg_response_time:.2f}s")
        
        print(f"\n📊 Results by Category:")
        for cat, metrics in report.category_results.items():
            print(f"\n   {cat.upper()}:")
            print(f"      Questions: {metrics['count']}")
            print(f"      Exact Match: {metrics['avg_exact_match']:.4f}")
            print(f"      F1 Score: {metrics['avg_f1_score']:.4f}")
            print(f"      Response Time: {metrics['avg_response_time']:.2f}s")
        
        print(f"\n{'='*70}\n")
    
    def save_report(self, report: BenchmarkReport, output_dir: str = "./benchmark_results"):
        """Save benchmark report to file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.benchmark_name}_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # Convert to dict
        report_dict = {
            'benchmark_name': report.benchmark_name,
            'benchmark_type': report.benchmark_type,
            'total_questions': report.total_questions,
            'avg_exact_match': report.avg_exact_match,
            'avg_f1_score': report.avg_f1_score,
            'avg_semantic_similarity': report.avg_semantic_similarity,
            'avg_response_time': report.avg_response_time,
            'category_results': report.category_results,
            'system_config': report.system_config,
            'timestamp': report.timestamp,
            'individual_results': [asdict(r) for r in report.individual_results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Report saved to: {filepath}")
        
        # Also save a summary CSV
        csv_filepath = filepath.with_suffix('.csv')
        self._save_csv_summary(report, csv_filepath)
        print(f"💾 CSV summary saved to: {csv_filepath}")
    
    def _save_csv_summary(self, report: BenchmarkReport, filepath: Path):
        """Save summary results as CSV"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'question_id', 'category', 'exact_match', 'f1_score',
                'semantic_similarity', 'response_time', 'context_source'
            ])
            
            # Data
            for result in report.individual_results:
                writer.writerow([
                    result.question_id,
                    result.category,
                    result.exact_match,
                    result.f1_score,
                    result.semantic_similarity,
                    result.response_time,
                    result.context_source
                ])


# =======================
# USAGE EXAMPLE
# =======================
from agent_improved import EnhancedAgenticMemorySystem
if __name__ == "__main__":
    # This is a standalone example - in practice, import your actual system
    print("🔧 Benchmark System Ready")
    print("\nTo use with your Enhanced Agentic Memory System:\n")
    
    example_code = '''
from enhanced_agentic_memory_system import EnhancedAgenticMemorySystem
from agent_benchmark_system import BenchmarkRunner, LongBenchV2Loader, BenchmarkType

# 1. Initialize your system
# system = EnhancedAgenticMemorySystem()

# 2. Load benchmark questions
# Option A: From file
questions = LongBenchV2Loader.load_from_file("longbench_v2_data.json")

# Option B: From HuggingFace
questions = LongBenchV2Loader.load_from_huggingface("THUDM/LongBench")

# Option C: Sample benchmark
questions = LongBenchV2Loader.create_sample_benchmark()

# 3. Create benchmark runner
runner = BenchmarkRunner(system)

# 4. Run benchmark
report = runner.run_benchmark(
    questions=questions,
    benchmark_name="LongBench_v2_Test",
    benchmark_type=BenchmarkType.LONGBENCH_V2,
    ingest_contexts=True,
    verbose=True
)

# 5. Save results
runner.save_report(report, output_dir="./benchmark_results")

# 6. Access specific metrics
print(f"Overall F1 Score: {report.avg_f1_score:.4f}")
print(f"Category breakdown: {report.category_results}")
'''
    
    # 1. Initialize your system
    system = EnhancedAgenticMemorySystem()

    # 2. Load benchmark questions
    # Option A: From file
    
    questions = LongBenchV2Loader.load_from_file("data2.json")

    # Option B: From HuggingFace
    # questions = LongBenchV2Loader.load_from_huggingface("THUDM/LongBench-v2")
    # print(questions)

    # Option C: Sample benchmark
    # questions = LongBenchV2Loader.create_sample_benchmark()

    # 3. Create benchmark runner
    runner = BenchmarkRunner(system)

    # 4. Run benchmark
    report = runner.run_benchmark(
        questions=questions,
        benchmark_name="LongBench_v2_Test",
        benchmark_type=BenchmarkType.LONGBENCH_V2,
        ingest_contexts=True,
        verbose=True
    )

    # 5. Save results
    runner.save_report(report, output_dir="./benchmark_results")

    # 6. Access specific metrics
    print(f"Overall F1 Score: {report.avg_f1_score:.4f}")
    print(f"Category breakdown: {report.category_results}")




    # print(example_code)
    
    # Demo with sample questions
    print("\n" + "="*70)
    print("🎯 Running DEMO with sample questions")
    print("="*70 + "\n")
    
    # # Create sample questions
    # questions = LongBenchV2Loader.create_sample_benchmark()
    
    # print(f"Created {len(questions)} sample questions:")
    # for q in questions:
    #     print(f"  - {q.id}: {q.question[:50]}... [{q.category}]")
    
    # print("\n💡 To run a real benchmark, initialize your EnhancedAgenticMemorySystem")
    # print("   and use the code example shown above!")