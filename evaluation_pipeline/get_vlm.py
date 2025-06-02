"""
This module provides access to different Vision Language Models (VLMs) for image analysis.
Supported providers: Azure OpenAI, Groq, Ollama, HuggingFace
"""

import base64
import os
import requests
import json
from typing import Dict, List, Union, Optional, Any, Tuple, Callable, TypedDict
from enum import Enum
import importlib.util
import sys
import time

# Add parent directory to path to import from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Check if python-dotenv is installed and load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("python-dotenv package not found. Consider installing it with: pip install python-dotenv")

# Check if required packages are installed
required_packages = {
    "openai": "azure-openai",
    "groq": "groq",
    "ollama": "ollama"
}

for module, package in required_packages.items():
    if importlib.util.find_spec(module) is None:
        print(f"Package '{package}' not found. Consider installing it with: pip install {package}")



class VLMProvider(str, Enum):
    """Enum for supported VLM providers"""
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class VLMAnalysisType(str, Enum):
    """Type of analysis to perform on the image"""
    DESCRIPTION = "description"
    COUNTING = "counting"
    LOCALIZATION = "localization"
    IDENTIFICATION = "identification"
    COMPREHENSIVE = "comprehensive"
    QUESTION = "question"  # New type for direct questions


class VLMCallMetrics(TypedDict, total=False):
    """Metrics for a VLM call."""
    elapsed_time: float  # seconds
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    input_price: Optional[float]
    output_price: Optional[float]
    total_price: Optional[float]


class VLM:
    """
    Base class for Vision Language Models.
    
    Subclasses overriding analyze_image or analyze_image_from_base64 must call
    self._record_metrics(start_time, response) as the last step to ensure metrics are updated.
    """

    call_metrics: Optional[VLMCallMetrics]

    def __init__(self, model_name: str = None, api_key: str = None, **kwargs):
        """Initialize the VLM with provider-specific parameters"""
        self.model_name = model_name
        self.api_key = api_key
        self.extra_params = kwargs
        self.call_metrics: Optional[VLMCallMetrics] = None

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def encode_image_from_base64(self, base64_image: str) -> str:
        """Use an already encoded base64 image"""
        # If the image already contains the data URL prefix, return as is
        if base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
            return base64_image
        # Otherwise add a generic image prefix
        return f"data:image/jpeg;base64,{base64_image}"

    def analyze_image(self, image_path: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, **kwargs) -> dict:
        """Analyze image and return results based on analysis type"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def answer_question(self, image_path: str, question: str) -> str:
        """Answer a specific question about an image"""
        return self.analyze_image(image_path, VLMAnalysisType.QUESTION, question=question)

    
    def answer_question_from_base64(self, base64_image: str, question: str) -> str:
        """Answer a specific question using a base64-encoded image"""
        result = self.analyze_image_from_base64(base64_image, VLMAnalysisType.QUESTION, question=question)
        if isinstance(result, dict) and "result" in result:
            return self._extract_answer(result.get("result", ""))
        return ""
    
    def analyze_image_from_base64(self, base64_image: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, **kwargs) -> dict:
        """Analyze a base64-encoded image and return results"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _extract_answer(self, result: str) -> str:
        """Extract a concise answer from the result text"""
        # Remove common phrases like "The answer is" or "I count"
        result = result.strip()
        answer_prefixes = [
            "the answer is ", "there are ", "i count ", "i see ", 
            "the number is ", "the total is ", "the result is "
        ]
        
        # Try to find and remove common prefixes
        lower_result = result.lower()
        for prefix in answer_prefixes:
            if lower_result.startswith(prefix):
                result = result[len(prefix):]
                break
        
        # Remove periods and other punctuation at the end
        result = result.rstrip(".!,; ")
        
        return result


    def _get_token_counts_from_response(self, response: dict) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract input and output token counts from the response. Placeholder for provider-specific logic.

        Args:
            response: The response dict from the VLM call.

        Returns:
            Tuple of (input_tokens, output_tokens), or (None, None) if not available.
        """
        return None, None

    def _calculate_costs(
        self, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate input, output, and total price. Placeholder for provider-specific logic.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Tuple of (input_price, output_price, total_price), or (None, None, None) if not available.
        """
        return None, None, None


class AzureOpenAIVLM(VLM):
    """Azure OpenAI implementation for VLM"""
    
    def __init__(self, model_name: str = "gpt-4.1-mini", api_key: str = None, api_version: str = "2024-12-01-preview", 
                 endpoint: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_version = api_version
        self.endpoint = endpoint
        
        # Import openai library
        try:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            # Verify available models
            try:
                available_models = [model.id for model in self.client.models.list()]
                if self.model_name not in available_models:
                    print(f"Warning: Model '{self.model_name}' not found in available models.")
                    if available_models and any("gpt-4" in model for model in available_models):
                        gpt4_models = [model for model in available_models 
                                      if "gpt-4.1-mini" in model and "vision" in model.lower()]
                        if gpt4_models:
                            print(f"Automatically selecting first available GPT-4.1 vision model: {gpt4_models[0]}")
                            self.model_name = gpt4_models[0]
                        else:
                            gpt4_models = [model for model in available_models if "gpt-4.1-mini" in model]
                            if gpt4_models:
                                print(f"Warning: No GPT-4.1 vision models found. Trying: {gpt4_models[0]}")
                                self.model_name = gpt4_models[0]
            except Exception as e:
                print(f"Warning: Could not list available models. {str(e)}")
        except ImportError:
            print("Azure OpenAI package not installed. Install with: pip install azure-openai")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
    
    def generate_prompt(self, analysis_type: VLMAnalysisType, **kwargs) -> str:
        """Generate prompt based on analysis type"""
        if analysis_type == VLMAnalysisType.QUESTION and 'question' in kwargs:
            return f"Look at this image and answer the following question concisely: {kwargs['question']}"
            
        prompts = {
            VLMAnalysisType.DESCRIPTION: "Describe the scene in this image in detail.",
            VLMAnalysisType.COUNTING: "Count the number of distinct objects/pieces in this image and list them.",
            VLMAnalysisType.LOCALIZATION: "Describe the position of each object in this image, using coordinates or spatial relationships.",
            VLMAnalysisType.IDENTIFICATION: "Identify each object in this image, including their types, colors, and any distinguishing features.",
            VLMAnalysisType.COMPREHENSIVE: """Analyze this image comprehensively and provide:
1. A detailed description of the scene
2. The count of distinct objects/pieces
3. The position and spatial relationships between objects
4. Identification of each object with its features (type, color, etc.)"""
        }
        return prompts.get(analysis_type, prompts[VLMAnalysisType.COMPREHENSIVE])
    
    def _get_token_counts_from_response(self, response: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract input and output token counts from the Azure OpenAI response.
        Response structure is based on `response.usage` containing `prompt_tokens` and `completion_tokens`.

        Args:
            response: The response object from the Azure OpenAI API call.

        Returns:
            A tuple containing (input_tokens, output_tokens). Returns (None, None) if counts cannot be extracted.
        """
        try:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            return input_tokens, output_tokens
        except AttributeError:
            print("Warning: 'usage' attribute with 'prompt_tokens' or 'completion_tokens' not found in Azure OpenAI response.")
            return None, None
        except Exception as e:
            print(f"Warning: Error extracting token counts from Azure OpenAI response: {str(e)}")
            return None, None

    def _calculate_costs(
        self, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate input, output, and total price for Azure OpenAI models.
        
        TODO: Verify and update with Azure-specific pricing for self.model_name.
        Current placeholders are based on OpenAI's public pricing for similar models.

        Args:
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (completion) tokens.

        Returns:
            A tuple (input_price, output_price, total_price). Returns (None, None, None) if costs cannot be calculated.
        """
        cost_per_input_token: Optional[float] = None
        cost_per_output_token: Optional[float] = None

        # Using "gpt-4o-mini" pricing as a reference for "gpt-4.1-mini"
        # Input: $0.15 / 1M tokens => 0.00000015 per token
        # Output: $0.60 / 1M tokens => 0.0000006 per token
        # These need to be confirmed for Azure's specific pricing for the model in use.
        if "gpt-4.1-mini" in self.model_name or "gpt-4o-mini" in self.model_name: # Approximate check
            cost_per_input_token = 0.4 / 1_000_000
            cost_per_output_token = 1.60 / 1_000_000
        elif "gpt-4.1" in self.model_name: # A broader GPT-4 category, potentially Turbo
            # Example for GPT-4 Turbo (illustrative)
            # Input: $10.00 / 1M tokens => 0.00001 per token
            # Output: $30.00 / 1M tokens => 0.00003 per token
            cost_per_input_token = 2.00 / 1_000_000
            cost_per_output_token = 8.00 / 1_000_000
        # Add other model pricing tiers as necessary

        if input_tokens is not None and output_tokens is not None and \
           cost_per_input_token is not None and cost_per_output_token is not None:
            input_price = input_tokens * cost_per_input_token
            output_price = output_tokens * cost_per_output_token
            total_price = input_price + output_price
            return input_price, output_price, total_price
        
        return None, None, None
    
    def analyze_image(self, image_path: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, 
                     **kwargs) -> dict:
        """Analyze image using Azure OpenAI"""
        base64_image = self.encode_image(image_path)
        return self.analyze_image_from_base64(base64_image, analysis_type, **kwargs)
    
    def analyze_image_from_base64(self, base64_image: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, **kwargs) -> dict:
        """Analyze a base64-encoded image using Azure OpenAI"""
        prompt = self.generate_prompt(analysis_type, **kwargs)
        # Ensure base64_image has the correct prefix
        if not base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
            base64_image = f"data:image/jpeg;base64,{base64_image}"
        
        try:
            # print(f"Using model: {self.model_name}")
            start_time = time.time() # Capture start time
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes images in detail."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]}
                ],
                max_tokens=1000  # TODO: Consider making max_tokens configurable
            )

            elapsed_time = time.time() - start_time
            input_tokens, output_tokens = self._get_token_counts_from_response(response)
            input_price, output_price, total_price = self._calculate_costs(input_tokens, output_tokens)
            current_metrics: VLMCallMetrics = {"elapsed_time": elapsed_time}
            if input_tokens is not None:
                current_metrics["input_tokens"] = input_tokens
            if output_tokens is not None:
                current_metrics["output_tokens"] = output_tokens
            if input_price is not None:
                current_metrics["input_price"] = input_price
            if output_price is not None:
                current_metrics["output_price"] = output_price
            if total_price is not None:
                current_metrics["total_price"] = total_price
            
            self.call_metrics = current_metrics
            return {
                "provider": VLMProvider.AZURE_OPENAI,
                "model": self.model_name,
                "analysis_type": analysis_type,
                "result": response.choices[0].message.content,
                "call_metrics": self.call_metrics
            }
        except Exception as e:
            elapsed_time_on_error = time.time() - start_time
            self.call_metrics = VLMCallMetrics(elapsed_time=elapsed_time_on_error)
            # You might want to add the error to call_metrics if VLMCallMetrics supports it
            # or log it more formally.
            return {"error": str(e), "call_metrics": self.call_metrics}


class GroqVLM(VLM):
    """Groq implementation for VLM"""

    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", api_key: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)

        # Import groq library
        try:
            import groq
            self.client = groq.Groq(api_key=self.api_key)
            
            # Test connection and get available models
            try:
                models = self.client.models.list()
                self.available_models = [model.id for model in models]
                print(f"Available Groq models: {', '.join(self.available_models)}")
                
                # Check if the requested model exists
                if self.model_name not in self.available_models:
                    print(f"Warning: Model '{self.model_name}' not found in available models.")
                    # Try to select a suitable model
                    if any("llama-4" in model.lower() for model in self.available_models):
                        llama_models = [model for model in self.available_models if "llama-4" in model.lower()]
                        print(f"Using alternative model: {llama_models[0]}")
                        self.model_name = llama_models[0]
            except Exception as e:
                print(f"Warning: Could not list Groq models: {str(e)}")
                
        except ImportError:
            print("Groq package not installed. Install with: pip install groq")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")

    def generate_prompt(self, analysis_type: VLMAnalysisType, **kwargs) -> str:
        """Generate prompt based on analysis type"""
        if analysis_type == VLMAnalysisType.QUESTION and 'question' in kwargs:
            return f"Look at this image and answer the following question concisely: {kwargs['question']}"
            
        prompts = {
            VLMAnalysisType.DESCRIPTION: "Describe what you see in this image in detail.",
            VLMAnalysisType.COUNTING: "Count the number of distinct objects/pieces in this image and list them.",
            VLMAnalysisType.LOCALIZATION: "Describe the position of each object in this image, using coordinates or spatial relationships.",
            VLMAnalysisType.IDENTIFICATION: "Identify each object in this image, including their types, colors, and any distinguishing features.",
            VLMAnalysisType.COMPREHENSIVE: """Analyze this image comprehensively and provide:
                                            1. A detailed description of the scene
                                            2. The count of distinct objects/pieces
                                            3. The position and spatial relationships between objects
                                            4. Identification of each object with its features (type, color, etc.)"""
        }
        return prompts.get(analysis_type, prompts[VLMAnalysisType.COMPREHENSIVE])
    
    def _get_token_counts_from_response(self, response: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract input and output token counts from the Groq response.
        Assumes response.usage contains prompt_tokens and completion_tokens.

        Args:
            response: The response object from the Groq API call.

        Returns:
            A tuple containing (input_tokens, output_tokens). Returns (None, None) if counts cannot be extracted.
        """
        try:
            # Groq's API usually includes an 'x_groq' object with usage details in the response,
            # or directly a 'usage' object similar to OpenAI.
            # Check for response.usage first
            if hasattr(response, 'usage') and response.usage is not None:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                return input_tokens, output_tokens
            # Check for response.x_groq.usage as another common pattern
            elif hasattr(response, 'x_groq') and response.x_groq is not None and hasattr(response.x_groq, 'usage') and response.x_groq.usage is not None:
                input_tokens = response.x_groq.usage.prompt_tokens
                output_tokens = response.x_groq.usage.completion_tokens
                return input_tokens, output_tokens
            else:
                # Fallback for responses where usage might be directly in the choices (less common for Groq directly but good to check)
                if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'usage'):
                     usage = response.choices[0].usage
                     if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                         return usage.prompt_tokens, usage.completion_tokens
                print("Warning: Token usage information not found in the expected Groq response structure.")
                return None, None
        except AttributeError:
            print("Warning: 'usage' attribute with 'prompt_tokens' or 'completion_tokens' not found in Groq response.")
            return None, None
        except Exception as e:
            print(f"Warning: Error extracting token counts from Groq response: {str(e)}")
            return None, None

    def _calculate_costs(
        self, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate input, output, and total price for Groq models.
        
        For Groq, cost is explicitly set to None as their pricing is often based on compute time (LPUs) 
        or includes generous free tiers not directly tied to per-token costs in the same way as other providers.
        Token counts can still be useful for understanding usage patterns.

        Args:
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (completion) tokens.

        Returns:
            A tuple (None, None, None) for (input_price, output_price, total_price).
        """
        # Groq's pricing model is different (often LPU-based or large free tiers).
        # Explicitly returning None for costs as per user request.
        return None, None, None
    
    def analyze_image(self, image_path: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, 
                    **kwargs) -> dict:
        """Analyze image using Groq"""
        base64_image = self.encode_image(image_path)
        return self.analyze_image_from_base64(base64_image, analysis_type, **kwargs)
        
    def analyze_image_from_base64(self, base64_image: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE,**kwargs) -> dict:
        """Analyze a base64-encoded image using Groq"""
        start_time = time.time()
        prompt = self.generate_prompt(analysis_type, **kwargs)
        
        # Ensure base64_image has the correct prefix
        if not base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
            base64_image = f"data:image/jpeg;base64,{base64_image}"
        
        try:
            # Use multimodal approach with image
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes images in detail."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]}
                ],
                max_tokens=1000
            )
            
            elapsed_time = time.time() - start_time
            input_tokens, output_tokens = self._get_token_counts_from_response(response)
            input_price, output_price, total_price = self._calculate_costs(input_tokens, output_tokens)

            current_metrics: VLMCallMetrics = {"elapsed_time": elapsed_time}
            if input_tokens is not None: current_metrics["input_tokens"] = input_tokens
            if output_tokens is not None: current_metrics["output_tokens"] = output_tokens
            if input_price is not None: current_metrics["input_price"] = input_price
            if output_price is not None: current_metrics["output_price"] = output_price
            if total_price is not None: current_metrics["total_price"] = total_price
            
            self.call_metrics = current_metrics
            return {
                "provider": VLMProvider.GROQ,
                "model": self.model_name,
                "analysis_type": analysis_type,
                "result": response.choices[0].message.content,
                "call_metrics": self.call_metrics
            }
        except Exception as e:
            elapsed_time_on_error = time.time() - start_time
            error_message = str(e)
            print(f"Error with Groq image analysis: {error_message}")
            
            # Fallback to text-only approach if image processing fails
            if "image" in error_message.lower() or "multimodal" in error_message.lower():
                print("Falling back to text-only approach as this Groq model might not support images")
                # Note: The fallback itself will set its own call_metrics
                return self._text_only_fallback(image_path=None, analysis_type=analysis_type, base64_image=base64_image, **kwargs)
            
            self.call_metrics = VLMCallMetrics(elapsed_time=elapsed_time_on_error)
            return {"error": error_message, "call_metrics": self.call_metrics}
    
    def _text_only_fallback(self, image_path: str = None, analysis_type: VLMAnalysisType = None, base64_image: str = None, **kwargs) -> dict:
        """Fallback to text-only approach if image processing is not supported"""
        start_time_fallback = time.time()
        prompt = self.generate_prompt(analysis_type, **kwargs)
        
        # Read image and get basic info
        try:
            from PIL import Image
            import io
            
            # Get basic image information
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        format_type = img.format
                        mode = img.mode
                        
                        # Add image metadata to the prompt
                        image_info = f"\nImage information: {width}x{height} pixels, {format_type} format, {mode} mode"
                        full_prompt = f"{prompt}\n{image_info}\n\nPlease analyze this image based on the above information."
                except Exception as img_error:
                    print(f"Warning: Could not read image metadata: {str(img_error)}")
                    full_prompt = prompt
            else:
                full_prompt = prompt
                
        except ImportError:
            print("Warning: PIL/Pillow not installed. Install with: pip install Pillow")
            full_prompt = prompt
            
        print("Note: Using text-only approach for Groq analysis.")
            
        try:
            # Use text-only approach
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps with image analysis based on descriptions."},
                    {"role": "user", "content": f"I have an image to analyze. {full_prompt}"}
                ],
                max_tokens=1000
            )
            
            elapsed_time_fallback = time.time() - start_time_fallback
            input_tokens_fb, output_tokens_fb = self._get_token_counts_from_response(response)
            input_price_fb, output_price_fb, total_price_fb = self._calculate_costs(input_tokens_fb, output_tokens_fb)

            fallback_metrics: VLMCallMetrics = {"elapsed_time": elapsed_time_fallback}
            if input_tokens_fb is not None: fallback_metrics["input_tokens"] = input_tokens_fb
            if output_tokens_fb is not None: fallback_metrics["output_tokens"] = output_tokens_fb
            if input_price_fb is not None: fallback_metrics["input_price"] = input_price_fb
            if output_price_fb is not None: fallback_metrics["output_price"] = output_price_fb
            if total_price_fb is not None: fallback_metrics["total_price"] = total_price_fb
            
            # Set self.call_metrics to the fallback metrics when this path is taken
            self.call_metrics = fallback_metrics
            print(f"Groq Call metrics (text_fallback): {self.call_metrics}")
            
            return {
                "provider": VLMProvider.GROQ,
                "model": self.model_name,
                "analysis_type": analysis_type,
                "result": response.choices[0].message.content,
                "note": "Used text-only fallback approach as image processing wasn't supported.",
                "call_metrics": self.call_metrics
            }
        except Exception as e:
            elapsed_time_on_error_fb = time.time() - start_time_fallback
            # Set self.call_metrics to the fallback error metrics
            self.call_metrics = VLMCallMetrics(elapsed_time=elapsed_time_on_error_fb)
            return {"error": str(e), "call_metrics": self.call_metrics}


class OllamaVLM(VLM):
    """Ollama implementation for VLM"""

    def __init__(self, model_name: str = "llava:latest", host: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, api_key=None, **kwargs)
        self.host = host

        # Import ollama library if available
        try:
            import ollama
            self.client = ollama
        except ImportError:
            print("Ollama package not installed. Install with: pip install ollama")
        except Exception as e:
            print(f"Error initializing Ollama client: {e}")

    def generate_prompt(self, analysis_type: VLMAnalysisType, **kwargs) -> str:
        """Generate prompt based on analysis type"""
        if analysis_type == VLMAnalysisType.QUESTION and 'question' in kwargs:
            return f"Look at this image and answer the following question concisely: {kwargs['question']}"
            
        prompts = {
            VLMAnalysisType.DESCRIPTION: "Describe the scene in this image in detail.",
            VLMAnalysisType.COUNTING: "Count the number of distinct objects/pieces in this image and list them.",
            VLMAnalysisType.LOCALIZATION: "Describe the position of each object in this image, using coordinates or spatial relationships.",
            VLMAnalysisType.IDENTIFICATION: "Identify each object in this image, including their types, colors, and any distinguishing features.",
            VLMAnalysisType.COMPREHENSIVE: """Analyze this image comprehensively and provide:
                                            1. A detailed description of the scene
                                            2. The count of distinct objects/pieces
                                            3. The position and spatial relationships between objects
                                            4. Identification of each object with its features (type, color, etc.)"""
        }
        return prompts.get(analysis_type, prompts[VLMAnalysisType.COMPREHENSIVE])

    def analyze_image(self, image_path: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, 
                    **kwargs) -> dict:
        """Analyze image using Ollama"""
        prompt = self.generate_prompt(analysis_type, **kwargs)
        start_time = time.time()

        try:
            with open(image_path, "rb") as image_file:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    images=[image_path],
                    options={
                        "temperature": 0.2,
                    }
                )
            
            elapsed_time = time.time() - start_time
            input_tokens = response.get("prompt_eval_count")
            output_tokens = response.get("eval_count")

            current_metrics: VLMCallMetrics = {
                "elapsed_time": elapsed_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_price": None,
                "output_price": None,
                "total_price": None
            }
            self.call_metrics = current_metrics
            
            return {
                "provider": VLMProvider.OLLAMA,
                "model": self.model_name,
                "analysis_type": analysis_type,
                "result": response["response"],
                "call_metrics": self.call_metrics
            }
        except Exception as e:
            elapsed_time_on_error = time.time() - start_time
            self.call_metrics = VLMCallMetrics(
                elapsed_time=elapsed_time_on_error,
                input_price=None, output_price=None, total_price=None
            )
            return {"error": str(e), "call_metrics": self.call_metrics}
            
    def analyze_image_from_base64(self, base64_image: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE,
                                **kwargs) -> dict:
        """Analyze a base64-encoded image using Ollama"""
        prompt = self.generate_prompt(analysis_type, **kwargs)
        start_time = time.time()
        
        try:
            # Save the base64 image temporarily to a file
            import tempfile
            # import base64 # Already imported at module level

            # Remove data URL prefix if present
            if base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
                # Extract just the base64 part
                if 'jpeg' in base64_image:
                    base64_image_data = base64_image.replace('data:image/jpeg;base64,', '')
                else:
                    base64_image_data = base64_image.replace('data:image/png;base64,', '')
            else:
                base64_image_data = base64_image # Assume it's already just the data
            
            decoded_image_data = base64.b64decode(base64_image_data)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(decoded_image_data)
                temp_path = temp_file.name
            
            # Process with the temporary file
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    images=[temp_path],
                    options={
                        "temperature": 0.2,
                    }
                )
                # print(f"Ollama response: {response}") # Keep for debugging if needed

                elapsed_time = time.time() - start_time
                input_tokens = response.get("prompt_eval_count")
                output_tokens = response.get("eval_count")

                current_metrics: VLMCallMetrics = {
                    "elapsed_time": elapsed_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_price": None,
                    "output_price": None,
                    "total_price": None
                }
                self.call_metrics = current_metrics
                result = {
                    "provider": VLMProvider.OLLAMA,
                    "model": self.model_name,
                    "analysis_type": analysis_type,
                    "result": response["response"],
                    "call_metrics": self.call_metrics
                }
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except Exception as unlink_e: # More specific exception handling if desired
                    print(f"Warning: Could not delete temporary file {temp_path}: {unlink_e}")
                    
            return result
        except Exception as e:
            elapsed_time_on_error = time.time() - start_time
            self.call_metrics = VLMCallMetrics(
                elapsed_time=elapsed_time_on_error,
                input_price=None, output_price=None, total_price=None
            )
            return {"error": str(e), "call_metrics": self.call_metrics}


class HuggingFaceVLM(VLM):
    """HuggingFace implementation for VLM"""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", api_key: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Import huggingface_hub if available (for model info)
        try:
            from huggingface_hub import HfApi
            self.hf_api = HfApi(token=api_key)
            print(f"Connected to HuggingFace API with model: {model_name}")
        except ImportError:
            print("huggingface_hub package not installed. Install with: pip install huggingface_hub")
            self.hf_api = None
        except Exception as e:
            print(f"Error initializing HuggingFace API: {e}")
            self.hf_api = None

    def generate_prompt(self, analysis_type: VLMAnalysisType, **kwargs) -> str:
        """Generate prompt based on analysis type"""
        if analysis_type == VLMAnalysisType.QUESTION and 'question' in kwargs:
            return f"Look at this image and answer: {kwargs['question']}"
            
        prompts = {
            VLMAnalysisType.DESCRIPTION: "Describe this image in detail.",
            VLMAnalysisType.COUNTING: "Count all objects in this image.",
            VLMAnalysisType.LOCALIZATION: "Locate all objects in this image.",
            VLMAnalysisType.IDENTIFICATION: "Identify all objects in this image.",
            VLMAnalysisType.COMPREHENSIVE: "Analyze this image comprehensively."
        }
        return prompts.get(analysis_type, prompts[VLMAnalysisType.COMPREHENSIVE])

    def analyze_image(self, image_path: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE, 
                     **kwargs) -> dict:
        """Analyze image using HuggingFace"""
        base64_image = self.encode_image(image_path)
        return self.analyze_image_from_base64(base64_image, analysis_type, **kwargs)
        
    def analyze_image_from_base64(self, base64_image: str, analysis_type: VLMAnalysisType = VLMAnalysisType.COMPREHENSIVE,
                                 **kwargs) -> dict:
        """Analyze a base64-encoded image using HuggingFace"""
        prompt = self.generate_prompt(analysis_type, **kwargs)
        
        # Remove data URL prefix if present
        if base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
            if 'jpeg' in base64_image:
                base64_image = base64_image.replace('data:image/jpeg;base64,', '')
            else:
                base64_image = base64_image.replace('data:image/png;base64,', '')
        
        try:
            # Prepare payload based on model capabilities
            if "/blip-" in self.model_name.lower() or "image-to-text" in self.model_name.lower():
                # Image captioning models
                payload = {"inputs": {"image": base64_image}}
            elif "/vilt-" in self.model_name.lower() or "visual-question-answering" in self.model_name.lower():
                # VQA models
                payload = {
                    "inputs": {
                        "image": base64_image,
                        "question": kwargs.get("question", prompt)
                    }
                }
            else:
                # General multimodal models
                payload = {
                    "inputs": base64_image,
                    "parameters": {
                        "prompt": prompt,
                        "max_new_tokens": 250,
                        "temperature": 0.2
                    }
                }

            # Make API call
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Process response - format varies based on model type
            response_json = response.json()
            
            # Handle different response formats
            if isinstance(response_json, list) and len(response_json) > 0:
                if isinstance(response_json[0], dict) and "generated_text" in response_json[0]:
                    result = response_json[0]["generated_text"]
                elif isinstance(response_json[0], str):
                    result = response_json[0]
                else:
                    result = str(response_json[0])
            elif isinstance(response_json, dict):
                if "generated_text" in response_json:
                    result = response_json["generated_text"]
                else:
                    result = str(response_json)
            else:
                result = str(response_json)
                
            return {
                "provider": VLMProvider.HUGGINGFACE,
                "model": self.model_name,
                "analysis_type": analysis_type,
                "result": result
            }
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                return {"error": "Rate limit exceeded. Please wait or upgrade your HuggingFace account."}
            elif "404" in str(e):
                return {"error": f"Model '{self.model_name}' not found on HuggingFace."}
            elif "401" in str(e):
                return {"error": "Unauthorized. Check your HuggingFace API token."}
            else:
                return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}


class VLMQuestionHandler:
    """Handler for processing questions with VLMs"""
    
    def __init__(self, provider: VLMProvider = None, **kwargs):
        """
        Initialize the VLM Question Handler.
        
        Args:
            provider: VLM provider to use. If None, will auto-select based on available credentials
            **kwargs: Additional arguments to pass to the VLM provider
        """
        if provider is None:
            # Auto-detect provider based on environment variables
            if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
                provider = VLMProvider.AZURE_OPENAI
            elif os.environ.get("GROQ_API_KEY"):
                provider = VLMProvider.GROQ
            elif os.environ.get("HUGGINGFACE_API_KEY"):
                provider = VLMProvider.HUGGINGFACE
            else:
                # Default to Ollama as it's local and doesn't require API keys
                provider = VLMProvider.OLLAMA
            
            print(f"Auto-selected VLM provider: {provider.value}")
        
        # Initialize the VLM
        self.vlm = get_vlm(provider, **kwargs)
        self.provider = provider
    
    def process_questions(self, image_path: str, questions: List[str]) -> List[str]:
        """
        Process a list of questions about an image using the VLM.
        
        Args:
            image_path: Path to the image file
            questions: List of questions to ask about the image
            
        Returns:
            List of answers from the VLM
        """
        answers = []
        
        # Check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process each question
        for question in questions:
            answer = self.vlm.answer_question(image_path, question)
            answers.append(answer)
            
        return answers
    
    def process_questions_from_base64(self, base64_image: str, questions: List[str]) -> List[str]:
        """
        Process a list of questions about a base64-encoded image.
        
        Args:
            base64_image: Base64-encoded image string
            questions: List of questions to ask about the image
            
        Returns:
            List of answers from the VLM
        """
        answers = []
        
        # Process each question
        for question in questions:
            answer = self.vlm.answer_question_from_base64(base64_image, question)
            answers.append(answer)
            
        return answers


def get_vlm(provider: VLMProvider, **kwargs) -> VLM:
    """
    Factory function to get the appropriate VLM based on provider

    Args:
        provider: The VLM provider to use
        **kwargs: Provider-specific arguments

    Returns:
        An initialized VLM instance
    """
    providers = {
        VLMProvider.AZURE_OPENAI: AzureOpenAIVLM,
        VLMProvider.GROQ: GroqVLM,
        VLMProvider.OLLAMA: OllamaVLM,
        VLMProvider.HUGGINGFACE: HuggingFaceVLM
    }

    if provider not in providers:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: {', '.join([p.value for p in VLMProvider])}")

    return providers[provider](**kwargs)


# Example usage
if __name__ == "__main__":
    print("Testing VLM functionality...")

    # Define the images directory path
    images_dir = "/home/test_count"

    # Check if the directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory not found at {images_dir}")
        # Try to find any image in the current directory as fallback
        test_image_path = os.path.join(
            os.path.dirname(__file__), "test_image.jpg")
        if not os.path.exists(test_image_path):
            print("Looking for images in the current directory...")
            for file in os.listdir(os.path.dirname(__file__)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    test_image_path = os.path.join(
                        os.path.dirname(__file__), file)
                    print(f"Found image: {test_image_path}")
                    break
            else:
                print("No image files found. Please provide an image for testing.")
                test_image_path = input(
                    "Enter the path to an image file for testing: ")
    else:
        # Find all image files in the specified directory
        image_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            print(f"No image files found in {images_dir}")
            test_image_path = input(
                "Enter the path to an image file for testing: ")
        else:
            # Print found images with index numbers
            print(f"Found {len(image_files)} images in {images_dir}:")
            for i, img_path in enumerate(image_files):
                print(f"[{i}] {os.path.basename(img_path)}")

            # Let the user choose an image or use the first one
            selection = input(
                f"Select an image (0-{len(image_files)-1}) or press Enter to use the first one: ")
            if selection.strip() and selection.isdigit() and 0 <= int(selection) < len(image_files):
                test_image_path = image_files[int(selection)]
            else:
                test_image_path = image_files[0]

            print(f"Using image: {test_image_path}")

    # Let the user select the provider to test
    print("\nSelect a provider to test:")
    print("[0] Azure OpenAI")
    print("[1] Groq")
    print("[2] Ollama")
    print("[3] HuggingFace")
    provider_selection = input("Enter your choice (0-3) or press Enter for Azure OpenAI: ")

    selected_provider = VLMProvider.AZURE_OPENAI
    if provider_selection == "1":
        selected_provider = VLMProvider.GROQ
    elif provider_selection == "2":
        selected_provider = VLMProvider.OLLAMA
    elif provider_selection == "3":
        selected_provider = VLMProvider.HUGGINGFACE

    print(f"Testing with provider: {selected_provider.value}")

    # Ask user for analysis type
    print("\nAvailable analysis types:")
    for i, analysis_type in enumerate(VLMAnalysisType):
        print(f"[{i}] {analysis_type.value}")

    analysis_selection = input(f"Select analysis type (0-{len(VLMAnalysisType)-1}) or press Enter for comprehensive analysis: ")
    if analysis_selection.strip() and analysis_selection.isdigit() and 0 <= int(analysis_selection) < len(VLMAnalysisType):
        selected_analysis = list(VLMAnalysisType)[int(analysis_selection)]
    else:
        selected_analysis = VLMAnalysisType.COMPREHENSIVE

    print(f"Using analysis type: {selected_analysis.value}")

    # Create output directory for results if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    # Test the selected provider
    if selected_provider == VLMProvider.AZURE_OPENAI:
        # Azure OpenAI testing logic
        try:
            # Get credentials from environment variables
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

            # Try to find a suitable model
            model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4.1-mini")
            if not model_name:
                print(
                    "No model specified in AZURE_OPENAI_MODEL_NAME, will attempt to detect automatically")
                model_name = "gpt-4.1-mini"  # Default to GPT-4 if not specified

            if not api_key or not endpoint:
                print("Azure OpenAI credentials not found in environment variables.")
                print("Please ensure you have the following in your .env file:")
                print("AZURE_OPENAI_API_KEY=your_api_key")
                print("AZURE_OPENAI_ENDPOINT=your_endpoint_url")
                print("AZURE_OPENAI_MODEL_NAME=your_model_name (optional)")
                print(
                    "AZURE_OPENAI_API_VERSION=your_api_version (optional, defaults to 2024-12-01-preview)")
            else:
                vlm = get_vlm(
                    VLMProvider.AZURE_OPENAI,
                    api_key=api_key,
                    endpoint=endpoint,
                    model_name=model_name,
                    api_version=api_version
                )

                if os.path.exists(test_image_path):
                    print(f"Analyzing image: {test_image_path}")

                    # Start analysis
                    print("\nStarting image analysis. This may take a moment...")
                    try:
                        result = vlm.analyze_image(
                            test_image_path,
                            analysis_type=selected_analysis
                        )

                        if "error" in result:
                            print(f"Error during analysis: {result['error']}")
                        else:
                            print("\nAnalysis Result:")
                            print("-" * 50)
                            print(f"Provider: {result['provider']}")
                            print(f"Model: {result['model']}")
                            print(f"Analysis Type: {result['analysis_type']}")
                            print("-" * 50)
                            print(result['result'])
                            print("-" * 50)

                            # Save result to file with timestamp and image name
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_name = os.path.splitext(
                                os.path.basename(test_image_path))[0]
                            result_file = os.path.join(output_dir, f"{timestamp}_{image_name}_{selected_analysis.value}.json")

                            with open(result_file, "w") as f:
                                json.dump(result, f, indent=2)
                            print(f"Result saved to {result_file}")

                    except Exception as image_error:
                        print(f"Error analyzing image: {str(image_error)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Image file not found: {test_image_path}")
        except Exception as e:
            print(f"Error testing Azure OpenAI: {str(e)}")
            import traceback
            traceback.print_exc()

    elif selected_provider == VLMProvider.GROQ:
        try:
            # Get credentials from environment variables
            api_key = os.environ.get("GROQ_API_KEY")
            model_name = os.environ.get("GROQ_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

            if not api_key:
                print("Groq credentials not found in environment variables.")
                print("Please ensure you have the following in your .env file:")
                print("GROQ_API_KEY=your_api_key")
                print("GROQ_MODEL_NAME=your_model_name (optional, defaults to meta-llama/llama-4-scout-17b-16e-instruct)")
                exit(1)

            print(f"Testing Groq with model: {model_name}")

            # Initialize the Groq VLM
            vlm = get_vlm(
                VLMProvider.GROQ,
                api_key=api_key,
                model_name=model_name
            )

            if os.path.exists(test_image_path):
                print(f"Analyzing image: {test_image_path}")

                # Start analysis
                print("\nStarting image analysis with Groq. This may take a moment...")

                try:
                    result = vlm.analyze_image(
                        test_image_path,
                        analysis_type=selected_analysis
                    )

                    if "error" in result:
                        print(f"Error during analysis: {result['error']}")

                        # Try to list available models in Groq
                        try:
                            import groq
                            client = groq.Groq(api_key=api_key)
                            models = client.models.list()
                            print("\nAvailable models in Groq:")
                            model_ids = []
                            for model in models:
                                # Handle different model list return formats
                                if hasattr(model, 'id'):
                                    model_ids.append(model.id)
                                elif isinstance(model, dict) and 'id' in model:
                                    model_ids.append(model['id'])
                                elif isinstance(model, tuple) and len(model) > 0:
                                    model_ids.append(str(model[0]))
                            
                            for model_id in model_ids:
                                print(f"- {model_id}")

                            print("\nPlease use one of these models in your .env file as GROQ_MODEL_NAME.")
                            print("Note: Groq doesn't currently support direct image analysis for all models.")
                        except Exception as list_error:
                            print(f"Error listing Groq models: {str(list_error)}")
                            print("Please verify your Groq API key.")
                    else:
                        print("\nAnalysis Result:")
                        print("-" * 50)
                        print(f"Provider: {result['provider']}")
                        print(f"Model: {result['model']}")
                        print(f"Analysis Type: {result['analysis_type']}")
                        if "note" in result:
                            print(f"Note: {result['note']}")
                        print("-" * 50)
                        print(result['result'])
                        print("-" * 50)

                        # Save result to file with timestamp and image name
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_name = os.path.splitext(os.path.basename(test_image_path))[0]
                        result_file = os.path.join(output_dir, f"{timestamp}_{image_name}_{selected_analysis.value}_groq.json")

                        with open(result_file, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"Result saved to {result_file}")

                except Exception as image_error:
                    print(f"Error analyzing image with Groq: {str(image_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Image file not found: {test_image_path}")

        except Exception as e:
            print(f"Error setting up Groq: {str(e)}")
            import traceback
            traceback.print_exc()

    elif selected_provider == VLMProvider.OLLAMA:
        try:
            # Get configuration from environment variables
            model_name = os.environ.get("OLLAMA_MODEL_NAME", "llava:latest")
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

            print(f"Testing Ollama with model: {model_name} at {host}")

            # Initialize the Ollama VLM
            vlm = get_vlm(
                VLMProvider.OLLAMA,
                model_name=model_name,
                host=host
            )

            if os.path.exists(test_image_path):
                print(f"Analyzing image: {test_image_path}")

                # Start analysis
                print("\nStarting image analysis with Ollama. This may take a moment...")

                try:
                    result = vlm.analyze_image(
                        test_image_path,
                        analysis_type=selected_analysis
                    )

                    if "error" in result:
                        print(f"Error during analysis: {result['error']}")
                        print("\nPlease ensure Ollama is running and has the required model.")
                        print("Run 'ollama pull llava' to get the latest LLaVA model.")
                    else:
                        print("\nAnalysis Result:")
                        print("-" * 50)
                        print(f"Provider: {result['provider']}")
                        print(f"Model: {result['model']}")
                        print(f"Analysis Type: {result['analysis_type']}")
                        print("-" * 50)
                        print(result['result'])
                        print("-" * 50)

                        # Save result to file with timestamp and image name
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_name = os.path.splitext(os.path.basename(test_image_path))[0]
                        result_file = os.path.join(output_dir, f"{timestamp}_{image_name}_{selected_analysis.value}.json")

                        with open(result_file, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"Result saved to {result_file}")

                except Exception as image_error:
                    print(f"Error analyzing image with Ollama: {str(image_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Image file not found: {test_image_path}")

        except Exception as e:
            print(f"Error setting up Ollama: {str(e)}")
            import traceback
            traceback.print_exc()

    elif selected_provider == VLMProvider.HUGGINGFACE:
        try:
            # Get credentials from environment variables
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
            model_name = os.environ.get("HUGGINGFACE_MODEL_NAME", "Salesforce/blip-image-captioning-large")

            print(f"Testing HuggingFace with model: {model_name}")

            # Initialize the HuggingFace VLM
            vlm = get_vlm(
                VLMProvider.HUGGINGFACE,
                api_key=api_key,
                model_name=model_name
            )

            if os.path.exists(test_image_path):
                print(f"Analyzing image: {test_image_path}")

                # Start analysis
                print("\nStarting image analysis with HuggingFace. This may take a moment...")

                try:
                    result = vlm.analyze_image(
                        test_image_path,
                        analysis_type=selected_analysis
                    )

                    if "error" in result:
                        print(f"Error during analysis: {result['error']}")
                        print("\nSuggested HuggingFace models for image analysis:")
                        print("- Salesforce/blip-image-captioning-large (image captioning)")
                        print("- dandelin/vilt-b32-finetuned-vqa (visual question answering)")
                        print("- microsoft/git-large-coco (image captioning)")
                    else:
                        print("\nAnalysis Result:")
                        print("-" * 50)
                        print(f"Provider: {result['provider']}")
                        print(f"Model: {result['model']}")
                        print(f"Analysis Type: {result['analysis_type']}")
                        print("-" * 50)
                        print(result['result'])
                        print("-" * 50)

                        # Save result to file with timestamp and image name
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_name = os.path.splitext(os.path.basename(test_image_path))[0]
                        result_file = os.path.join(output_dir, f"{timestamp}_{image_name}_{selected_analysis.value}_hf.json")

                        with open(result_file, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"Result saved to {result_file}")

                except Exception as image_error:
                    print(f"Error analyzing image with HuggingFace: {str(image_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Image file not found: {test_image_path}")

        except Exception as e:
            print(f"Error setting up HuggingFace: {str(e)}")
            import traceback
            traceback.print_exc()
