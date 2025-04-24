from flask import Flask, render_template, request
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from functools import lru_cache
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter

app = Flask(__name__)

# Configuration
MODEL_NAME = "deepset/roberta-large-squad2"
ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity

@app.context_processor
def inject_vars():
    return {
        'torch': torch,
        'MODEL_NAME': MODEL_NAME,
        'cuda_available': torch.cuda.is_available()
    }

@lru_cache(maxsize=1)
def get_qa_pipeline():
    """Cache the pipeline to avoid reloading"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        handle_impossible_answer=True
    )

def preprocess_text(text):
    """Clean and normalize input text"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def postprocess_answer(answer):
    """Clean up model answers"""
    answer = answer.strip()
    if answer and answer[-1] not in {'.', '!', '?'}:
        answer += '.'
    return answer

def chunk_passage(passage, tokenizer, max_tokens=384, stride=128):
    """Improved chunking with sentence awareness and overlap"""
    sentences = [s.strip() for s in passage.split('. ') if s.strip()]
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        tokens = tokenizer(current_chunk + " " + sentence, 
                         return_tensors="pt", 
                         truncation=False)["input_ids"][0]
        
        if len(tokens) <= max_tokens:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    final_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, 
                         return_overflowing_tokens=True,
                         max_length=max_tokens,
                         stride=stride,
                         return_offsets_mapping=True,
                         truncation=True)
        
        for input_ids in inputs["input_ids"]:
            final_chunks.append(tokenizer.decode(input_ids, skip_special_tokens=True))
    
    return final_chunks

def calculate_metrics(answers, question, passage):
    """Calculate comprehensive answer quality metrics"""
    if not answers:
        return {}
    
    scores = [ans["score"] for ans in answers]
    answer_texts = [ans["text"] for ans in answers]
    
    metrics = {
        "confidence": {
            "average": np.mean(scores),
            "highest": max(scores),
            "lowest": min(scores),
            "std_dev": np.std(scores)
        },
        "length": {
            "chars": np.mean([len(ans) for ans in answer_texts]),
            "words": np.mean([len(ans.split()) for ans in answer_texts])
        }
    }
    
    if len(answer_texts) > 1:
        embeddings = ST_MODEL.encode(answer_texts)
        sim_matrix = util.cos_sim(embeddings, embeddings)
        metrics["semantic_similarity"] = {
            "average": float(np.mean(sim_matrix)),
            "min": float(np.min(sim_matrix)),
            "max": float(np.max(sim_matrix))
        }
    
    passage_length = len(passage)
    metrics["position"] = {
        "normalized_start": np.mean([ans["start"]/passage_length for ans in answers]),
        "normalized_end": np.mean([ans["end"]/passage_length for ans in answers])
    }
    
    all_words = ' '.join(answer_texts).split()
    word_counts = Counter(all_words)
    metrics["redundancy"] = {
        "unique_words": len(word_counts),
        "total_words": len(all_words),
        "repeat_rate": sum(cnt > 1 for cnt in word_counts.values())/len(word_counts)
    }
    
    return metrics

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = datetime.now()
        passage = preprocess_text(request.form.get("passage", ""))
        question = preprocess_text(request.form.get("question", ""))
        
        if not passage or not question:
            return render_template("index.html", error="Please enter both passage and question.")
        
        try:
            qa_pipeline = get_qa_pipeline()
            passage_chunks = chunk_passage(passage, qa_pipeline.tokenizer)
            
            answers = []
            chunk_times = []
            
            for chunk in passage_chunks:
                chunk_start = datetime.now()
                try:
                    result = qa_pipeline({"question": question, "context": chunk})
                    if result["score"] > 0.01:
                        answers.append({
                            "text": result["answer"],
                            "score": result["score"],
                            "start": result["start"],
                            "end": result["end"]
                        })
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                chunk_times.append((datetime.now() - chunk_start).total_seconds())
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not answers:
                return render_template("index.html", 
                                     answer="No confident answer found.",
                                     metrics={},
                                     passage=passage,
                                     question=question,
                                     processing_time=processing_time)
            
            answers.sort(key=lambda x: x["score"], reverse=True)
            unique_answers = {}
            for ans in answers:
                clean_text = ans["text"].strip().rstrip('.')
                if clean_text not in unique_answers or ans["score"] > unique_answers[clean_text]["score"]:
                    unique_answers[clean_text] = ans
            
            metrics = calculate_metrics(answers, question, passage)
            metrics["processing"] = {
                "total_time": processing_time,
                "avg_chunk_time": np.mean(chunk_times) if chunk_times else 0,
                "chunks_processed": len(passage_chunks)
            }
            
            final_answer = " ".join(unique_answers.keys())
            final_answer = postprocess_answer(final_answer)
            
            return render_template("index.html", 
                                answer=final_answer,
                                metrics=metrics,
                                passage=passage,
                                question=question)
            
        except Exception as e:
            print(f"System error: {e}")
            return render_template("index.html", 
                                error="An error occurred while processing your request.",
                                passage=passage,
                                question=question)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)