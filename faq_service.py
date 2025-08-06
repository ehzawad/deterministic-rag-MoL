import os
import json
import hashlib
import asyncio
import logging
import traceback
import glob
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Import Bengali text normalizer
from bengali_normalizer import normalize_bengali, normalize_for_matching

# Load configuration
def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        # Fallback configuration
        return {
            "models": {
                "embedding_model": "text-embedding-3-large",
                "core_model": "gpt-4o-mini"
            },
            "system": {
                "confidence_threshold": 0.9,
                "max_candidates": 1,
                "embedding_dimensions": 1024
            },
            "directories": {
                "faq_dir": "faq_data",
                "cache_dir": "cache"
            },
            "logging": {
                "level": "INFO"
            }
        }

# Load configuration
config = load_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LandRecord-FAQ-Service')

# Constants from config
FAQ_DIR = config['directories']['faq_dir']
CACHE_DIR = config['directories']['cache_dir']
CHROMA_DB_DIR = os.path.join(CACHE_DIR, "chroma_db")
FILE_HASH_CACHE = os.path.join(CACHE_DIR, "file_hashes.json")
CONFIDENCE_THRESHOLD = 0.0  # Threshold for matching
MAX_CANDIDATES = config['system']['max_candidates']
EMBEDDING_MODEL = config['models']['embedding_model']
EMBEDDING_DIMENSIONS = config['system']['embedding_dimensions']

# Removed complex prime word routing - using semantic similarity across all collections

# Each text file becomes its own collection
# Collection names will be based on filename without extension

class LandRecordFAQService:
    """Core Land Record FAQ Service using ChromaDB with file-as-cluster routing"""
    
    def __init__(self):
        # Handle missing API key gracefully
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.test_mode = False
        else:
            self.client = None
            self.test_mode = True
            logger.warning("OPENAI_API_KEY not found. Running in test mode (no embeddings).")
        
        self.file_hashes = {}
        self.initialized = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Removed complex hybrid matcher
        
        # Load file hashes
        self._load_file_hashes()
    
    def _load_file_hashes(self):
        """Load cached file hashes"""
        if os.path.exists(FILE_HASH_CACHE):
            try:
                with open(FILE_HASH_CACHE, 'r') as f:
                    self.file_hashes = json.load(f)
                logger.info(f"Loaded {len(self.file_hashes)} cached file hashes")
            except Exception as e:
                logger.error(f"Error loading cached file hashes: {e}")
                self.file_hashes = {}
    
    def _save_file_hashes(self):
        """Save file hashes to cache"""
        try:
            with open(FILE_HASH_CACHE, 'w') as f:
                json.dump(self.file_hashes, f)
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")
    
    def _discover_faq_files(self) -> List[str]:
        """Dynamically discover all .txt files in the FAQ directory"""
        try:
            if not os.path.exists(FAQ_DIR):
                logger.warning(f"FAQ directory '{FAQ_DIR}' does not exist.")
                return []
            
            txt_files = glob.glob(os.path.join(FAQ_DIR, "*.txt"))
            filenames = [os.path.basename(filepath) for filepath in txt_files]
            
            logger.info(f"Discovered {len(filenames)} .txt files in {FAQ_DIR}: {filenames}")
            return filenames
        except Exception as e:
            logger.error(f"Error discovering FAQ files: {e}")
            return []
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
                return hashlib.md5(file_data).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def _check_for_updates(self) -> Tuple[bool, Set[str]]:
        """Check if FAQ files have been modified since last run"""
        files_to_process = set()
        
        discovered_files = self._discover_faq_files()
        
        for filename in discovered_files:
            filepath = os.path.join(FAQ_DIR, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Warning: File {filepath} does not exist.")
                continue
                
            current_hash = self._calculate_file_hash(filepath)
            if current_hash:
                if filename not in self.file_hashes or self.file_hashes[filename] != current_hash:
                    files_to_process.add(filename)
                    self.file_hashes[filename] = current_hash
        
        return len(files_to_process) > 0, files_to_process
    
    def _preprocess_faq_file(self, filepath: str) -> List[Dict[str, str]]:
        """Preprocess FAQ file to extract Q&A pairs with Bengali text handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            q_markers = ["Question:", "প্রশ্ন:"]
            a_markers = ["Answer:", "উত্তর:"]
            
            pairs = []
            current_question = None
            current_answer = []
            in_answer = False
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if not line:
                    continue
                
                is_question_line = any(line.startswith(marker) for marker in q_markers)
                is_answer_line = any(line.startswith(marker) for marker in a_markers)
                
                if is_question_line:
                    if current_question and current_answer:
                        clean_q = self._clean_text(current_question)
                        clean_a = ' '.join(current_answer)
                        if clean_q and clean_a:
                            pairs.append({
                                "question": clean_q,
                                "answer": clean_a,
                                "source": os.path.basename(filepath)
                            })
                    
                    for marker in q_markers:
                        if line.startswith(marker):
                            current_question = line[len(marker):].strip()
                            break
                    
                    current_answer = []
                    in_answer = False
                
                elif is_answer_line:
                    for marker in a_markers:
                        if line.startswith(marker):
                            answer_text = line[len(marker):].strip()
                            if answer_text:
                                current_answer.append(answer_text)
                            break
                    
                    in_answer = True
                
                elif in_answer and line:
                    current_answer.append(line)
                
                elif not in_answer and line:
                    if current_question:
                        current_question += " " + line
                    else:
                        current_question = line
            
            if current_question and current_answer:
                clean_q = self._clean_text(current_question)
                clean_a = ' '.join(current_answer)
                if clean_q and clean_a:
                    pairs.append({
                        "question": clean_q,
                        "answer": clean_a,
                        "source": os.path.basename(filepath)
                    })
            
            logger.info(f"Extracted {len(pairs)} Q&A pairs from {filepath}")
            return pairs
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Bengali text using comprehensive normalizer"""
        if not text:
            return text
        
        # Apply Bengali text normalization
        text = normalize_bengali(text)
        
        # Normalize common land record terms
        land_terms = {
            'e-porcha': 'ই-পর্চা',
            'mutation': 'মিউটেশন',
            'khatian': 'খতিয়ান',
            'porcha': 'পর্চা',
            'map': 'নকশা'
        }
        
        # Normalize common spelling variations and typos
        spelling_variations = {
            'নামজারী': 'নামজারি',
            'খারীজ': 'খারিজ',
            'মিউটেসন': 'মিউটেশন',
            'পর্চ্চা': 'পর্চা',
            'এডিসি': 'এডিসি',
            'প্রবাসি': 'প্রবাসী'
        }
        
        for eng_term, bengali_term in land_terms.items():
            text = text.replace(eng_term, bengali_term)
            
        for variation, standard in spelling_variations.items():
            text = text.replace(variation, standard)
        
        # Remove reference numbers (sequences of 8+ digits)
        text = re.sub(r'\b\d{8,}\b', '', text)
        
        return text.strip()
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using configured embedding model"""
        if not texts:
            return []
        
        if self.test_mode:
            # In test mode, return dummy embeddings
            logger.info(f"Test mode: Creating {len(texts)} dummy embeddings")
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSIONS
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []
    
    def _get_collection_name(self, filename: str) -> str:
        """Get ChromaDB collection name based on filename"""
        # Remove .txt extension and use filename as collection name
        base_name = filename.replace('.txt', '')
        return f"faq_{base_name}"
    
    def _update_collection(self, filename: str, faq_pairs: List[Dict[str, str]]):
        """Create/update collection for this specific file"""
        try:
            collection_name = self._get_collection_name(filename)
            
            # Delete existing collection if it exists
            try:
                existing_collections = [coll.name for coll in self.chroma_client.list_collections()]
                if collection_name in existing_collections:
                    self.chroma_client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete existing collection: {e}")
            
            # Create fresh collection for this file
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"filename": filename, "source_file": filename}
            )
            logger.info(f"Created collection: {collection_name}")
            
            if not faq_pairs:
                logger.warning(f"No FAQ pairs to add to collection {collection_name}")
                return
            
            # Prepare data for ChromaDB
            questions = [pair["question"] for pair in faq_pairs]
            embeddings = self._create_embeddings(questions)
            
            if len(embeddings) != len(questions):
                logger.error(f"Embedding count mismatch for {filename}")
                return
            
            # Safer batch addition - add in smaller chunks to prevent corruption
            batch_size = 50  # Smaller batches are more reliable
            
            for i in range(0, len(faq_pairs), batch_size):
                batch_end = min(i + batch_size, len(faq_pairs))
                batch_pairs = faq_pairs[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_questions = questions[i:batch_end]
                
                ids = [f"{filename}_{j}" for j in range(i, batch_end)]
                metadatas = [
                    {
                        "question": pair["question"],
                        "answer": pair["answer"], 
                        "source": pair["source"]
                    }
                    for pair in batch_pairs
                ]
                
                try:
                    collection.add(
                        ids=ids,
                        embeddings=batch_embeddings,
                        documents=batch_questions,
                        metadatas=metadatas
                    )
                    logger.debug(f"Added batch {i//batch_size + 1} to {collection_name}")
                except Exception as batch_error:
                    logger.error(f"Error adding batch to {collection_name}: {batch_error}")
                    raise  # Re-raise to trigger collection rebuild
            
            logger.info(f"Added {len(faq_pairs)} entries to collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error updating collection for {filename}: {e}")
            logger.error(traceback.format_exc())
    
    
    def _detect_prime_words(self, query: str) -> List[str]:
        """Removed prime word detection - using single collection"""
        return []
    
    def _detect_prime_words_cached(self, query_lower: str) -> List[str]:
        """Removed prime word detection - using single collection"""
        return []
    
    def _test_mode_search(self, collection, query: str, n_results: int) -> List[Dict]:
        """Simple text-based search for test mode"""
        try:
            # Get all documents from collection
            all_data = collection.get(include=["metadatas", "documents"])
            
            if not all_data['metadatas']:
                return []
            
            candidates = []
            query_lower = query.lower()
            
            for metadata, document in zip(all_data['metadatas'], all_data['documents']):
                # Simple text matching score
                question_lower = metadata["question"].lower()
                
                # Count word matches
                query_words = set(query_lower.split())
                question_words = set(question_lower.split())
                
                if not query_words:
                    continue
                
                # Normalize both texts for exact matching
                query_normalized = normalize_for_matching(query)
                question_normalized = normalize_for_matching(metadata["question"])
                
                # Check for exact match first (should be 1.0)
                if query_normalized == question_normalized:
                    similarity = 1.0
                elif query_lower == question_lower:
                    similarity = 1.0  # Fallback to original comparison
                else:
                    # Calculate simple similarity score
                    matches = len(query_words.intersection(question_words))
                    similarity = matches / len(query_words) if query_words else 0
                    
                    # Boost exact substring matches
                    if query_lower in question_lower:
                        similarity += 0.5
                    
                    # Cap at 1.0
                    similarity = min(similarity, 1.0)
                
                if similarity > 0:  # Only include if there's some match
                    candidates.append({
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "source": metadata["source"],
                        "score": similarity,
                        "collection": collection.name
                    })
            
            # Sort by similarity and return top results
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:n_results]
            
        except Exception as e:
            logger.error(f"Error in test mode search: {e}")
            return []
    
    def _search_collection(self, collection_name: str, query: str, n_results: int = MAX_CANDIDATES) -> List[Dict]:
        """Search within a specific ChromaDB collection"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if self.test_mode:
                # In test mode, use simple text matching
                return self._test_mode_search(collection, query, n_results)
            
            query_embedding = self._create_embeddings([query])
            
            if not query_embedding:
                return []
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, collection.count()),
                include=["metadatas", "distances"]
            )
            
            candidates = []
            if results['metadatas'] and results['distances']:
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    # Convert distance to similarity (ChromaDB returns distances, we want similarity)
                    similarity = 1 - distance
                    
                    candidates.append({
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "source": metadata["source"],
                        "score": similarity,
                        "collection": collection.name
                    })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def _search_all_collections(self, query: str) -> List[Dict]:
        """Search through all file-based collections"""
        all_candidates = []
        
        try:
            collections = self.chroma_client.list_collections()
            logger.info(f"Searching {len(collections)} collections")
            
            for collection in collections:
                candidates = self._search_collection(collection.name, query, MAX_CANDIDATES)
                all_candidates.extend(candidates)
                logger.debug(f"Collection {collection.name}: found {len(candidates)} candidates")
                
        except Exception as e:
            logger.error(f"Error searching all collections: {e}")
        
        # Sort by similarity score and return top candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates[:MAX_CANDIDATES]
    
    def _llm_semantic_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Use LLM to semantically rerank candidates for better semantic understanding"""
        if not candidates or self.test_mode or not self.client:
            return candidates
        
        try:
            # Prepare context with query + top candidates (limit to top 8 for context efficiency)
            top_candidates = candidates[:8]
            context = f"Query: {query}\n\nFAQ Options:\n"
            
            for i, candidate in enumerate(top_candidates):
                context += f"{i+1}. Q: {candidate['question']}\n   A: {candidate['answer'][:200]}...\n   Source: {candidate['source']}\n\n"
            
            prompt = f"""You are a Bengali land record FAQ expert. Given this query and FAQ options, rank them by SEMANTIC SIMILARITY, not exact word matching.

CRITICAL Bengali Land Record Equivalencies:
- নামজারি = খারিজ = মিউটেশন (ALL mean land mutation/registration)
- খতিয়ান = পর্চা = রেকর্ড (land records variations)
- নকশা = ম্যাপ (map variations)
- ফি = টাকা = চার্জ = খরচ (fees/charges)
- অনলাইন = ই-পর্চা = ডিজিটাল (online services)
- স্ট্যাটাস = অবস্থা = ট্র্যাক (status checking)

PRIORITIZE SEMANTIC MEANING OVER EXACT WORDS.
Example: "নামজারি কত টাকা?" should match "মিউটেশনের ফি কত?" very highly.

{context}

Return ONLY the numbers (1-{len(top_candidates)}) in order of BEST SEMANTIC MATCH (meaning similarity), separated by commas.
Example: 3,1,5,2,4"""
            
            # Call LLM for semantic ranking with fallback models
            models_to_try = [
                config['models'].get('core_model', 'gpt-4o-mini'),
                'gpt-4o-mini'
            ]
            
            response = None
            for model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    logger.debug(f"Successfully used model: {model}")
                    break
                except Exception as model_error:
                    logger.debug(f"Model {model} failed: {model_error}")
                    continue
            
            if not response:
                logger.warning("All LLM models failed, falling back to original ranking")
                return candidates
            
            ranking_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM ranking response: {ranking_text}")
            
            # Parse the ranking
            try:
                rankings = [int(x.strip()) - 1 for x in ranking_text.split(',') if x.strip().isdigit()]
                
                # Reorder candidates based on LLM ranking
                reranked = []
                used_indices = set()
                
                for rank_idx in rankings:
                    if 0 <= rank_idx < len(top_candidates) and rank_idx not in used_indices:
                        reranked.append(top_candidates[rank_idx])
                        used_indices.add(rank_idx)
                
                # Add any remaining candidates
                for i, candidate in enumerate(top_candidates):
                    if i not in used_indices:
                        reranked.append(candidate)
                
                # Add remaining original candidates that weren't in top 8
                if len(candidates) > 8:
                    reranked.extend(candidates[8:])
                
                logger.info(f"LLM reranked {len(top_candidates)} candidates")
                return reranked
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing LLM ranking '{ranking_text}': {e}")
                return candidates
                
        except Exception as e:
            logger.error(f"Error in LLM semantic reranking: {e}")
            return candidates
    
    def _find_best_match(self, query: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Search through all collections and find best match"""
        cleaned_query = self._clean_text(query)
        
        # Search all file-based collections
        all_candidates = self._search_all_collections(query)
        
        # LLM Semantic Reranking for improved semantic understanding
        if all_candidates:
            logger.info(f"Applying LLM semantic reranking to {len(all_candidates)} candidates")
            all_candidates = self._llm_semantic_rerank(query, all_candidates)
        
        # Check if best match meets confidence threshold
        if all_candidates and all_candidates[0]['score'] >= CONFIDENCE_THRESHOLD:
            logger.info(f"Best match: {all_candidates[0]['question'][:50]}... (score: {all_candidates[0]['score']:.3f})")
            return all_candidates[0], all_candidates
        
        return None, all_candidates
    
    def initialize(self) -> bool:
        """Initialize the system by loading and preprocessing all FAQ data"""
        try:
            logger.info("Initializing Land Record FAQ Service...")
            
            if not os.path.exists(FAQ_DIR):
                logger.error(f"Error: FAQ directory '{FAQ_DIR}' does not exist.")
                return False
            
            discovered_files = self._discover_faq_files()
            if not discovered_files:
                logger.error("No .txt files found in FAQ directory.")
                return False
            
            # Check for updates
            updates_needed, files_to_process = self._check_for_updates()
            
            if not updates_needed:
                # Check if collections exist AND have data
                try:
                    collections = self.chroma_client.list_collections()
                    if collections:
                        total_entries = 0
                        
                        for collection in collections:
                            try:
                                coll = self.chroma_client.get_collection(collection.name)
                                count = coll.count()
                                total_entries += count
                            except Exception as e:
                                logger.warning(f"Error checking collection {collection.name}: {e}")
                                # Mark this collection for rebuilding if there's an issue
                                for filename, coll_type in FILE_TO_COLLECTION.items():
                                    if f"faq_{coll_type}" == collection.name:
                                        files_to_process.add(filename)
                                        break
                        
                        if total_entries > 0:
                            logger.info(f"No updates needed. Using existing ChromaDB collections with {total_entries} total entries.")
                            self.initialized = True
                            return True
                        else:
                            logger.info("Found existing collections but they are empty. Reprocessing all files.")
                            files_to_process = set(discovered_files)
                    else:
                        logger.info("No existing collections found, processing all files.")
                        files_to_process = set(discovered_files)
                        
                except Exception as e:
                    logger.warning(f"Error checking existing collections: {e}, processing all files.")
                    files_to_process = set(discovered_files)
            
            if not files_to_process:
                files_to_process = set(discovered_files)
            
            logger.info(f"Processing {len(files_to_process)} files...")
            
            # Process each file into its own collection
            for filename in files_to_process:
                filepath = os.path.join(FAQ_DIR, filename)
                if os.path.exists(filepath):
                    logger.info(f"Processing file: {filepath}")
                    faq_pairs = self._preprocess_faq_file(filepath)
                    if faq_pairs:
                        self._update_collection(filename, faq_pairs)
                    else:
                        logger.warning(f"No FAQ pairs extracted from {filepath}")
            
            # Save file hashes
            self._save_file_hashes()
            
            # Verify collections were created
            collections = self.chroma_client.list_collections()
            logger.info(f"Initialization complete. Created {len(collections)} collections.")
            self.initialized = True
            return len(collections) > 0
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def answer_query_async(self, query: str, debug: bool = False) -> Dict:
        """Answer a user query using the optimized routing system with hybrid matching"""
        if not self.initialized:
            logger.error("FAQ Service not initialized. Call initialize() first.")
            return {"found": False, "message": "System not initialized"}
        
        try:
            # Clean the query
            cleaned_query = self._clean_text(query)
            
            # Check if query is in Bengali
            has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
            
            # Find best match using routing with hybrid matching
            logger.info(f"Processing query: {cleaned_query}")
            best_match, all_candidates = self._find_best_match(query)  # Pass original query, not cleaned
            
            result = {"found": False, "confidence": 0.0}
            
            if best_match:
                result.update({
                    "found": True,
                    "matched_question": best_match["question"],
                    "answer": best_match["answer"],
                    "source": best_match["source"],
                    "confidence": best_match["score"],
                    "collection": best_match["collection"]
                })
                logger.info(f"Found match with confidence {best_match['score']:.3f} from {best_match['collection']}")
            else:
                # Return fallback message in appropriate language
                if has_bengali:
                    result["message"] = "দুঃখিত, আমি আপনার প্রশ্নের উত্তর খুঁজে পাইনি। অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লিখুন।"
                else:
                    result["message"] = "Sorry, I couldn't find an answer to your question. Please rephrase your question."
                
                if all_candidates:
                    result["confidence"] = all_candidates[0]["score"]
                    logger.info(f"Best candidate score {all_candidates[0]['score']:.3f} below threshold {CONFIDENCE_THRESHOLD}")
            
            # Add debug info if requested
            if debug:
                result["candidates"] = all_candidates[:5]  # Top 5 candidates
                result["threshold"] = CONFIDENCE_THRESHOLD
                result["total_collections"] = len(self.chroma_client.list_collections())
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            logger.error(traceback.format_exc())
            return {"found": False, "message": f"Error processing query: {str(e)}"}
    
    def answer_query(self, query: str, debug: bool = False) -> Dict:
        """Synchronous wrapper for answer_query_async"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.answer_query_async(query, debug)
            )
        finally:
            loop.close()
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            collections = self.chroma_client.list_collections()
            stats = {
                "total_collections": len(collections),
                "collections": {},
                "initialized": self.initialized,
                "test_mode": self.test_mode
            }
            
            for collection in collections:
                coll = self.chroma_client.get_collection(collection.name)
                stats["collections"][collection.name] = {
                    "count": coll.count(),
                    "metadata": collection.metadata
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict:
        """Simple health check for ChromaDB collections"""
        try:
            if not self.initialized:
                return {"status": "error", "message": "Service not initialized"}
            
            collections = self.chroma_client.list_collections()
            healthy_collections = []
            corrupted_collections = []
            
            for collection in collections:
                try:
                    coll = self.chroma_client.get_collection(collection.name)
                    count = coll.count()
                    
                    healthy_collections.append({
                        "name": collection.name,
                        "count": count
                    })
                    
                except Exception as e:
                    corrupted_collections.append({
                        "name": collection.name,
                        "error": str(e)
                    })
            
            status = "healthy" if not corrupted_collections else "degraded"
            
            return {
                "status": status,
                "healthy_collections": len(healthy_collections),
                "corrupted_collections": len(corrupted_collections),
                "details": {
                    "healthy": healthy_collections,
                    "corrupted": corrupted_collections
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {e}"}

# Global service instance - auto-initialize on import
faq_service = LandRecordFAQService()

# Auto-initialize the service when module is imported
logger.info("Auto-initializing Land Record FAQ Service...")
initialization_success = faq_service.initialize()

if initialization_success:
    logger.info("✅ Land Record FAQ Service initialized successfully!")
else:
    logger.error("❌ Land Record FAQ Service initialization failed!")

# Export the service instance
__all__ = ['faq_service'] 