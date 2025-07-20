import copy
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS, URL
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time
import gc


class RankGPT(BaseRanking):
    """
    Implements **RankGPT** with detailed GPU memory and performance logging.
    """

    def __init__(self, method: str = None, model_name: str = None, api_key: str = None , **kwargs):
        """
        Initializes a RankGPT instance.
        """
        self.method = method
        self.window_size = kwargs.get("window_size", 20) 
        self.step = kwargs.get("step", 10)
        self.endpoint = kwargs.get("endpoint", "https://api.openai.com/v1")
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        self.use_gpu = True
        self.use_bf16 = True
        
        # Performance tracking
        self.inference_times = []
        self.memory_snapshots = []
        
        if self.method == 'rankgpt-api':
            if model_name in URL:
                self.model_name = URL[model_name]['model_name']
                self.url = URL[model_name]['url']
            else:
                self.model_name = model_name
                self.url = self.endpoint
        else:
            self.model_name = model_name

        self._load(model_name)

    def _load(self, model_name: str = None) -> None:
        """
        Loads the GPT model and tokenizer with GPU optimizations.
        """
        if self.method == 'rankgpt-api':
            if model_name in URL:
                self.model = URL[model_name]['class'](self.api_key, self.url)
            else:
                self.model = URL['default']['class'](self.api_key, self.url)
        else:
            print(f"🚀 Loading {self.model_name} with GPU optimizations...")
            
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self._log_gpu_memory("Before Loading")
            
            # OPTIMIZED MODEL LOADING
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Force float16
                device_map="auto",          # Auto GPU placement
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,              # Fast tokenizer
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # VERIFY GPU PLACEMENT
            model_device = next(self.model.parameters()).device
            print(f"📍 Model loaded on: {model_device}")
            
            if 'cuda' not in str(model_device) and torch.cuda.is_available():
                #print("🚨 Forcing GPU placement...")
                self.model = self.model.cuda()
                #print(f"📍 Model now on: {next(self.model.parameters()).device}")
            
            # Show GPU memory usage after loading
            if torch.cuda.is_available():
                self._log_gpu_memory("After Loading")

    def _log_gpu_memory(self, stage="", detailed=False):
        """Log detailed GPU memory usage."""
        if not torch.cuda.is_available():
            print(f"   💻 {stage}: CPU mode")
            return
        
        # Force GPU synchronization for accurate measurements
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # if detailed:
        #     total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        #     free_memory = total_memory - reserved
        #     utilization = (allocated / total_memory) * 100
            
        #     print(f"   🎮 GPU Memory {stage}:")
        #     print(f"      Allocated: {allocated:.3f}GB")
        #     print(f"      Reserved:  {reserved:.3f}GB") 
        #     print(f"      Max Used:  {max_allocated:.3f}GB")
        #     print(f"      Free:      {free_memory:.3f}GB")
        #     print(f"      Usage:     {utilization:.1f}%")
        # else:
        #     print(f"   🎮 GPU Memory {stage}: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")
        
        return allocated, reserved

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Ranks documents with comprehensive logging.
        """
        print(f"\n🎯 Starting ranking for {len(documents)} documents")
        self._log_gpu_memory("Before Ranking", detailed=True)
        
        start_total = time.time()
        
        for doc_idx, document in enumerate(tqdm(documents, desc="Reranking Documents")):
            #print(f"\n📄 Document {doc_idx+1}/{len(documents)} - {len(document.contexts)} contexts")
            doc_start = time.time()
            
            reorder_contexts = self.sliding_windows(
                document, 
                rank_start=0, 
                rank_end=len(document.contexts), 
                window_size=self.window_size, 
                step=self.step
            )
            document.reorder_contexts = reorder_contexts
            
            doc_time = time.time() - doc_start
            #print(f"   ⏱️  Document {doc_idx+1} completed in {doc_time:.2f}s")
            
            # Memory cleanup and logging every document
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                self._log_gpu_memory(f"After Doc {doc_idx+1}")
            #break
        total_time = time.time() - start_total
        print(f"\n✅ All documents ranked in {total_time:.2f}s")
        self._print_performance_summary()
        
        return documents

    def sliding_windows(self, item=None, rank_start=0, rank_end=100, window_size=20, step=10):
        """
        Applies sliding window ranking with detailed logging.
        """
        #print(f"   🪟 Starting sliding windows: {rank_start}-{rank_end}, window={window_size}, step={step}")
        
        item.reorder_contexts = item.contexts
        item = copy.copy(item)
        item.reorder_contexts = item.contexts.copy()
        
        # Calculate total windows
        total_windows = 0
        temp_end = rank_end
        temp_start = rank_end - window_size
        while temp_start >= rank_start:
            temp_start = max(temp_start, rank_start)
            total_windows += 1
            temp_end = temp_end - step
            temp_start = temp_start - step
        
        #print(f"   📊 Will process {total_windows} windows")
        
        # Process windows with detailed logging
        window_num = 0
        end_pos = rank_end
        start_pos = rank_end - window_size
        
        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            window_num += 1
            
            #print(f"\n     🔹 Window {window_num}/{total_windows}: [{start_pos}:{end_pos}] ({end_pos-start_pos} items)")
            
            # Log memory before window processing
            mem_before_window = self._log_gpu_memory(f"Before Window {window_num}")
            window_start_time = time.time()
            
            item = self.permutation_pipeline(
                item=item, 
                rank_start=start_pos, 
                rank_end=end_pos,
                window_info=f"{window_num}/{total_windows}"
            )
            
            window_time = time.time() - window_start_time
            mem_after_window = self._log_gpu_memory(f"After Window {window_num}")
            
            # Calculate memory delta for this window
            # if torch.cuda.is_available() and mem_before_window and mem_after_window:
            #     mem_delta = mem_after_window[0] - mem_before_window[0]
            #     print(f"     ⏱️  Window {window_num} completed in {window_time:.3f}s (Δ{mem_delta:+.3f}GB)")
            # else:
            #     print(f"     ⏱️  Window {window_num} completed in {window_time:.3f}s")
            
            end_pos = end_pos - step
            start_pos = start_pos - step
            
        #print(f"   ✅ All {total_windows} windows completed")
        return item.reorder_contexts

    def permutation_pipeline(self, item=None, rank_start=0, rank_end=100, window_info=""):
        """
        Pipeline with detailed timing and memory logging.
        """
        # Create instruction phase
        inst_start = time.time()
        messages = self.create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
        inst_time = time.time() - inst_start
        
        # LLM inference phase with detailed logging
        llm_start = time.time()
        permutation = self.run_llm_logged(messages, window_info)
        llm_time = time.time() - llm_start
        
        # Process result phase  
        proc_start = time.time()
        item = self.receive_permutation(item=item, permutation=permutation, rank_start=rank_start, rank_end=rank_end)
        proc_time = time.time() - proc_start
        
        total_time = inst_time + llm_time + proc_time
        #print(f"       📊 Pipeline: Inst:{inst_time:.3f}s + LLM:{llm_time:.3f}s + Proc:{proc_time:.3f}s = {total_time:.3f}s")
        
        # Store performance data
        self.inference_times.append(llm_time)
        
        return item

    def run_llm_logged(self, messages, window_info=""):
        """
        LLM inference with detailed GPU memory and timing logging.
        """
        if self.method == 'rankgpt-api':
            return self.model.chat(model=self.model_name, messages=messages, temperature=0, return_text=True)
        
        # Pre-inference logging
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_before = torch.cuda.memory_allocated() / 1024**3
        
        with torch.no_grad():
            # Tokenization phase
            tokenize_start = time.time()
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    max_length=1024,        
                    truncation=True         
                )
                
                # Log tokenization results
                seq_len = input_ids.shape[1]
                tokenize_time = time.time() - tokenize_start
                
                # Device transfer phase
                transfer_start = time.time()
                model_device = next(self.model.parameters()).device
                input_ids = input_ids.to(model_device, non_blocking=True)
                transfer_time = time.time() - transfer_start
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_after_tokenize = torch.cuda.memory_allocated() / 1024**3
                    tokenize_mem_delta = gpu_after_tokenize - gpu_before
                else:
                    gpu_after_tokenize = 0
                    tokenize_mem_delta = 0
                
            except Exception as e:
                print(f"         ❌ Tokenization failed: {e}")
                return ""
            
            # Generation phase with detailed monitoring
            generate_start = time.time()
            try:
                terminators = [self.tokenizer.eos_token_id]
                eot_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_token is not None and eot_token != self.tokenizer.unk_token_id:
                    terminators.append(eot_token)
                
                # Memory snapshot before generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_before_gen = torch.cuda.memory_allocated() / 1024**3
                
                # GPU generation with mixed precision
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    responses = self.model.generate(
                        input_ids,
                        max_new_tokens=32,      
                        eos_token_id=terminators,
                        do_sample=False,        
                        temperature=1.0,
                        top_p=1.0,
                        use_cache=True,         
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_beams=1,           
                        early_stopping=True,
                        output_attentions=False,
                        output_hidden_states=False
                    )
                
                generate_time = time.time() - generate_start
                
                # Memory snapshot after generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_after_gen = torch.cuda.memory_allocated() / 1024**3
                    gen_mem_delta = gpu_after_gen - gpu_before_gen
                    gen_mem_peak = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    gpu_after_gen = 0
                    gen_mem_delta = 0
                    gen_mem_peak = 0
                
            except Exception as e:
                print(f"         ❌ Generation failed: {e}")
                return ""
            
            # Decoding phase
            decode_start = time.time()
            try:
                generated_tokens = responses[0][input_ids.shape[-1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decode_time = time.time() - decode_start
                
                # Final memory check
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_final = torch.cuda.memory_allocated() / 1024**3
                    total_mem_delta = gpu_final - gpu_before
                else:
                    gpu_final = 0
                    total_mem_delta = 0
                
            except Exception as e:
                print(f"         ❌ Decoding failed: {e}")
                return ""
        
        # Comprehensive logging
        total_time = tokenize_time + transfer_time + generate_time + decode_time
        tokens_per_sec = len(generated_tokens) / generate_time if generate_time > 0 else 0
        
        # print(f"         🔧 Detailed Breakdown {window_info}:")
        # print(f"            Input tokens: {seq_len}")
        # print(f"            Output tokens: {len(generated_tokens)}")
        # print(f"            Tokenize: {tokenize_time:.3f}s (Δ{tokenize_mem_delta:+.3f}GB)")
        # print(f"            Transfer: {transfer_time:.3f}s")
        # print(f"            Generate: {generate_time:.3f}s (Δ{gen_mem_delta:+.3f}GB, Peak:{gen_mem_peak:.3f}GB)")
        # print(f"            Decode: {decode_time:.3f}s")
        # print(f"            Total: {total_time:.3f}s (Δ{total_mem_delta:+.3f}GB)")
        # print(f"            Speed: {tokens_per_sec:.1f} tokens/sec")
        
        # Performance assessment
        # if generate_time < 0.5:
        #     perf_status = "🚀 EXCELLENT"
        # elif generate_time < 1.0:
        #     perf_status = "✅ GOOD"
        # elif generate_time < 2.0:
        #     perf_status = "⚠️ SLOW"
        # else:
        #     perf_status = "🐌 VERY SLOW"
        
        # print(f"            Performance: {perf_status}")
        
        return response

    def _print_performance_summary(self):
        """Print comprehensive performance summary."""
        if not self.inference_times:
            return
        
        times = self.inference_times
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # print(f"\n📈 PERFORMANCE SUMMARY:")
        # print(f"   🧠 LLM Inference Statistics:")
        # print(f"      Total inferences: {len(times)}")
        # print(f"      Average time: {avg_time:.3f}s")
        # print(f"      Fastest: {min_time:.3f}s")
        # print(f"      Slowest: {max_time:.3f}s")
        # print(f"      Std deviation: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.3f}s")
        
        # Throughput calculation
        total_inference_time = sum(times)
        throughput = len(times) / total_inference_time if total_inference_time > 0 else 0
        # print(f"      Throughput: {throughput:.2f} inferences/second")
        
        # Performance categorization
        # if avg_time > 3.0:
        #     print(f"   🚨 CRITICAL: Very slow performance - check GPU usage!")
        # elif avg_time > 1.5:
        #     print(f"   ⚠️ WARNING: Slow performance - optimization needed")
        # elif avg_time > 0.8:
        #     print(f"   ✅ ACCEPTABLE: Moderate performance")
        # else:
        #     print(f"   🚀 EXCELLENT: Fast performance!")
        
        # GPU memory summary
        # if torch.cuda.is_available():
        #     current_allocated = torch.cuda.memory_allocated() / 1024**3
        #     max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        #     total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
        #     print(f"   🎮 GPU Memory Summary:")
        #     print(f"      Current usage: {current_allocated:.2f}GB")
        #     print(f"      Peak usage: {max_allocated:.2f}GB")
        #     print(f"      Total available: {total_memory:.2f}GB")
        #     print(f"      Efficiency: {(max_allocated/total_memory)*100:.1f}% peak utilization")

    # Keep all existing methods unchanged
    def create_permutation_instruction(self, item=None, rank_start=0, rank_end=100):
        query = item.question.question
        num = len(item.contexts[rank_start: rank_end])
        max_length = 300
        messages = self.get_prefix_prompt(query, num)
        rank = 0
        for hit in item.contexts[rank_start: rank_end]:
            rank += 1
            content = hit.text
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            content = " ".join(content.split()[: int(max_length)])
            messages.append({"role": "user", "content": f"[{rank}] {content}"})
            messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
        messages.append({"role": "user", "content": self.get_post_prompt(query, num)})
        return messages

    def run_llm(self, messages):
        """Wrapper for backward compatibility."""
        return self.run_llm_logged(messages)

    def receive_permutation(self, item, permutation, rank_start=0, rank_end=100):
        response = self.clean_response(permutation)
        if not response.strip():
            return item
        try:
            response = [int(x) - 1 for x in response.split()]
        except ValueError:
            return item
        response = self.remove_duplicate(response)
        cut_range = item.contexts[rank_start:rank_end]
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            item.reorder_contexts[j + rank_start] = copy.copy(cut_range[x])
            if hasattr(item.reorder_contexts[j + rank_start], 'rank'):
                item.reorder_contexts[j + rank_start].rank = cut_range[j].rank
            if hasattr(item.reorder_contexts[j + rank_start], 'score'):
                item.reorder_contexts[j + rank_start].score = cut_range[j].score
        return item

    def get_prefix_prompt(self, query, num):
        return [{'role': 'system',
                 'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                {'role': 'user',
                 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    def clean_response(self, response: str):
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(self, response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def get_post_prompt(self, query, num):
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."