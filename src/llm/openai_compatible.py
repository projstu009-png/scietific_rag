from typing import List, Dict, Any, Optional
import requests

class OpenAICompatibleLLM:
    """OpenAI-compatible LLM client for local models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.api_key = config.get("api_key", "dummy")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"LLM generation failed: {response.text}")
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate with retrieved context"""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""Context:
{context_str}

Question: {query}

Answer the question based on the provided context. Be precise and cite sources using [1], [2], etc."""
        
        return self.generate(prompt, system_prompt)
