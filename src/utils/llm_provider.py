"""LLM provider wrapper with Ollama support (free, local)."""
import json
import logging
from typing import Dict, Optional, Any
import time

logger = logging.getLogger(__name__)


class LLMProviderWrapper:
    """Wrapper for LLM providers with fallback support."""
    
    def __init__(self, provider: str = "ollama", model: str = "llama3.2:3b", 
                 max_retries: int = 2, timeout: int = 30, api_key: Optional[str] = None):
        """
        Initialize LLM provider.
        
        Args:
            provider: "ollama" (free, local) or "groq" (free API)
            model: Model name (llama3.2:3b for ollama, llama-3.3-70b-versatile for groq)
            max_retries: Max retry attempts
            timeout: Request timeout in seconds
            api_key: API key for cloud providers (groq)
        """
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.total_calls = 0
        self.failed_calls = 0
        
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=model,
                    temperature=0.1,  # Low temperature for consistency
                    timeout=timeout
                )
                logger.info(f"Initialized Ollama with model {model}")
            except ImportError:
                raise ImportError("Install langchain-ollama: pip install langchain-ollama")
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}. Will use fallback.")
                self.llm = None
                
        elif provider == "groq":
            try:
                from langchain_groq import ChatGroq
                import os
                
                # Get API key from parameter or environment
                groq_api_key = api_key or os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY not found. Set via environment or parameter.")
                
                self.llm = ChatGroq(
                    model=model,
                    temperature=0.1,
                    groq_api_key=groq_api_key,
                    timeout=timeout
                )
                logger.info(f"Initialized Groq with model {model}")
            except ImportError:
                raise ImportError("Install langchain-groq: pip install langchain-groq")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
                raise
                
        else:
            raise ValueError(f"Provider {provider} not supported. Use 'ollama' or 'groq'.")
    
    def invoke_with_retry(self, prompt: str, response_format: str = "json") -> Optional[Dict[str, Any]]:
        """
        Invoke LLM with retry logic and validation.
        
        Args:
            prompt: Input prompt
            response_format: "json" or "text"
            
        Returns:
            Parsed response dict or None if failed
        """
        self.total_calls += 1
        
        if self.llm is None:
            logger.warning("LLM not available, returning None")
            self.failed_calls += 1
            return None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = self.llm.invoke(prompt)
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract content
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                logger.debug(f"LLM response ({latency_ms:.0f}ms): {content[:100]}")
                
                # Parse JSON if requested
                if response_format == "json":
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    try:
                        parsed = json.loads(content)
                        logger.info(f"LLM call successful (attempt {attempt + 1}, {latency_ms:.0f}ms)")
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
                        if attempt < self.max_retries:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                else:
                    return {"text": content}
                    
            except Exception as e:
                logger.warning(f"LLM invocation failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        
        # All retries failed
        self.failed_calls += 1
        logger.error(f"LLM call failed after {self.max_retries + 1} attempts")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        success_rate = ((self.total_calls - self.failed_calls) / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate
        }
