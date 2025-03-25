import requests
import os
import time
import pickle
import cohere
import json
from service.utils import compute_cost
import os
import anthropic
#from transformers import CodeGenTokenizerFast
#tokenizer_FFAI = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
from transformers import GPT2Tokenizer
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat # DocumentSchema, FunctionToolDefinition, ToolDefinition, ToolParameters
import google.generativeai as genai

import base64
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import logging
#genai.configure(api_key=os.environ["GEMINI_API_KEY"])


tokenizer_FFAI = GPT2Tokenizer.from_pretrained("gpt2")

class GenerationParameter(object):
    def __init__(self,
                 max_tokens=100,
                 temperature=0.1,
                 stop=["\n"],
                 date="20230301",
                 trial = 0,
                 ):
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.stop = stop
        self.date = date
        self.readable = dict()
        self.readable['max_tokens'] = max_tokens        
        self.readable['temperature'] = temperature
        self.readable['stop'] = stop
        self.readable['date'] = date
        #self.readable['trial'] = trial


    def get_dict(self,
                 ):
        return self.readable        
                  
class ModelService:
    """Base class for a model provider, currently we support API models, but this class can be subclassed with support
    for local GPU models"""
    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        """
            Given a context, generate a text output
        """        
        raise NotImplementedError

from pathlib import Path

class APIModelProvider(ModelService):
    """Provider that calculates conditional logprobs through a REST API"""
    # Use absolute path resolution
    _CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "serviceinfo.json"
    _CONFIG = json.load(open(_CONFIG_PATH))
    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        endpoint = self._get_endpoint()
        req = self._request_format(context,
                                   genparams)
        self.context = context
        if(use_save==True):
            #print("-try previous call-",savepath)
            try:
                response = self.read_response(savepath)
                #print('Have predicted before.')
                #print('prediction',self.mlservice.read_response(finalpath))
            except:
                #print("new call")
                response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
                self.write_response(savepath)
        else:
            time1 = time.time()
            response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
            latency = time.time()-time1

        result = self._response_format(response)
        cost = self._get_cost(context,result)        
        result['cost'] = cost
        result['latency'] = latency

        return result

    def read_response(self, path="test"):
        f = open(path, 'rb')
        self.response = pickle.load(f)
        #self._response2pred(self.response)
        f.close()
        try:
            return self.response.json()
        except:
            return self.response
            
    def write_response(self,path="test"):
        filehandler = open(path, 'wb') 
        pickle.dump(self.response, filehandler)
        filehandler.close()
        try:
            return self.response.json()
        except:
            return self.response
        
    def _request_format(self,
                     genparams):
        raise NotImplementedError

    def _response_format(self,
                         response,
                         ):
        raise NotImplementedError
         
    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for i in range(retries):
            #print("data is",data)
            #'''
            '''
            res = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=data,
                timeout=60)
            print("res is",res)
            print("made it")
            '''
            while True:
                try:
                    res = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=data,
                timeout=60)
                    res.raise_for_status()  # Raise an exception if the response is an HTTP error
                    break  # If we got a response, exit the loop
                except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                    # If we got a timeout or another kind of error, retry the request
                    print("timeout, retry")
                    continue
            #'''
            #print("res is", res)
            if res.status_code == 200:
                #print("succeed!")
                self.response = res
                return res.json()
            else:
                print("failed res is:",res)
            print(f"API call failed with {res}. Waiting {retry_grace_time} seconds")
            time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!") 

    def _get_cost(self,
                  context,
                  completion,
                ):
        tk1, tk2 = self._get_io_tokens(context,completion)
        cost = compute_cost(tk1,tk2,self._CONFIG[self._NAME][self._model])
        return cost
    
    def _get_endpoint(self):
        endpoint = self._ENDPOINT.format(engine=self._model)
        return endpoint
        
class OpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/engines/{engine}/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "openai"
    
    def __init__(self, model):
        self._model = model
        self.temp_cmp = {'raw':{'usage':{'prompt_tokens':0}}}
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _request_format(self,
                        context,
                        genparams):
        #tk1,tk2 = self._get_io_tokens(context,self.temp_cmp)
        if(self._model in ['tƒd']):
            #tk1 = len(context) # TODO: change this to real token count
            tk1 = len(tokenizer_FFAI(context)['input_ids'])
            print("length:",tk1)
            if(tk1+genparams.max_tokens>=2047):
                print("warning: the length is too large. only take the last few ones")
                context = context[-2048+genparams.max_tokens:]
        req = {
            "prompt": context,
            "echo": False,
            "max_tokens": genparams.max_tokens,
            "logprobs": 1,
            "temperature": genparams.temperature,
            "top_p": 1,
            "stop":genparams.stop,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result["completion"] = response['choices'][0]['text']
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        tk1 = completion['raw']['usage']['prompt_tokens']
        try:
            tk2 = completion['raw']['usage']['completion_tokens']
        except:
            tk2 = 0
            
        return tk1, tk2  
    
class OpenAIChatModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAICHAT_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "openaichat"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "messages": [{"content":context,"role":"user"}],
            #"echo": False,
            "max_tokens": genparams.max_tokens,
            #"logprobs": 1,
            "temperature": genparams.temperature,
            #"top_p": 1,
            #"stop":, # seems it does not allow \n to be the stopping token.
            "model":self._model,
            #"role":"user",
        }
        #print("the request is---")
        #print(req)
        return req
    
    def _response_format(self,
                         response,
                         ):
        #print("response is",response)
        result = dict()
        result['raw'] = response
        result["completion"] = response['choices'][0]['message']['content']
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        tk1 = completion['raw']['usage']['prompt_tokens']
        try:
            tk2 = completion['raw']['usage']['completion_tokens']
        except:
            tk2 = 0
            
        return tk1, tk2  

class AI21ModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("AI21_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete")
    _API_KEY = os.environ.get('AI21_STUDIO_API_KEY', None)
    _NAME = "ai21"
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AI21_STUDIO_API_KEY env var for running through AI21 Studio"
        self.client = AI21Client(api_key=os.environ.get("AI21_STUDIO_API_KEY"))

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "maxTokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "stopSequences":genparams.stop,
            "model":self._model,

        }           
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = completion['raw'].usage.prompt_tokens
        tk2 = completion['raw'].usage.completion_tokens
        return tk1, tk2 

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        while(1):
            try:
                response = self.client.chat.completions.create(
		model=data['model'],
		messages=[
			ChatMessage(
				role="user",
				content=data['prompt'],
			)
		],
		documents=[],
		tools=[],
		n=1,
		max_tokens=data['maxTokens'],
		temperature=data['temperature'],
		top_p=1,
		stop=data['stopSequences'],
		response_format=ResponseFormat(type="text"),
)
                return response 
            except Exception as e:
                print(f"Failed with errors {e}, retry")
                time.sleep(retry_grace_time)
  





class CohereAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("COHERE_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete")
    _API_KEY = os.environ.get('COHERE_STUDIO_API_KEY', None)
    _NAME = "cohere"

    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set COHERE_STUDIO_API_KEY env var for running through AI21 Studio"
    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        
        co = cohere.Client(api_key)
        try:
            response = co.generate( 
            model=self._model, 
            prompt=data['prompt'], 
            max_tokens=data['max_tokens'], 
            temperature=data['temperature'], 
            k=data['k'], 
            p=data['p'], 
            frequency_penalty=data['frequency_penalty'], 
            presence_penalty=data['presence_penalty'], 
            stop_sequences=data['stop_sequences'], 
            return_likelihoods=data['return_likelihoods']) 
        except:
            response = co.generate( 
            model=self._model, 
            prompt="test", 
            max_tokens=data['max_tokens'], 
            temperature=data['temperature'], 
            k=data['k'], 
            p=data['p'], 
            frequency_penalty=data['frequency_penalty'], 
            presence_penalty=data['presence_penalty'], 
            stop_sequences=data['stop_sequences'], 
            return_likelihoods=data['return_likelihoods']) 
            
        self.response = response
        return response

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "model":self._model,
            #"topKReturn": 10,
             "max_tokens":genparams.max_tokens, 
             "temperature":genparams.temperature, 
            "k":1,
            "num_generations":1,
            "p":1, 
            "frequency_penalty":0, 
            "presence_penalty":0, 
            "stop_sequences":genparams.stop, 
            "return_likelihoods":'ALL',          
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        fullresponse = dict()
        fullresponse['raw'] = ''

        try:
            #print("response is:", response)
            #print("type of the response",type(response))
            token_likelihoods = response.generations[0].token_likelihoods
            text = [i1.token for i1 in token_likelihoods]
            text = "".join(text)
            #print("text is",text[len(context):])
            fullresponse["completion"] = response.generations[0].text
        except:
            fullresponse["completion"]=''
        #print("completion:",fullresponse["completion"])
        return fullresponse
    
    def _get_io_tokens(self,context, completion):
        tk1 = len(context)/1000
        tk2 = len(completion)/1000
        return tk1, tk2
	
class ForeFrontAIModelProvider(APIModelProvider):
    _API_KEY = os.environ.get('FOREFRONT_API_KEY', None)  
    _NAME = "ffai"
    _ENDPOINT_MAP = {"QA":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
                     "CodeGen":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
                     "Pythia":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/pythia-20b/completions/vanilla",
                     }
    def __init__(self, model="QA"):
        self._model = model
        assert self._API_KEY is not None, "Please set FOREFRONT_API_KEY env var for running through Forefront"  
    
    def _request_format(self,
                        context,
                        genparams):
#        if(self._model in ['QA','text-babbage-001','text-ada-001']):
        if(1):
            #tk1 = len(context) # TODO: change this to real token count
            tk1 = len(tokenizer_FFAI(context)['input_ids'])
            print("length:",tk1)
            if(tk1+genparams.max_tokens>=2047):
                print("warning: the length is too large. only take the last few ones")
                context = context[-2048+genparams.max_tokens:]
        
        req = {
            "text": context,
            "numResults": 1,
            "length": genparams.max_tokens,
            "topKReturn": 1,
            "temperature":genparams.temperature,
            "stop":["\n"],
            "logprobs":1,
            "echo":False,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result["raw"] = response
        result['completion'] = response['result'][0]['completion']
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = len(tokenizer_FFAI(context)['input_ids'])
        tk2 = len(completion['raw']["logprobs"]['tokens'])
        return tk1, tk2
    
    def _get_endpoint(self):
        endpoint = self._ENDPOINT_MAP[self._model]
        return endpoint
			  
class TextSynthModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("TEXTSYNTH_ENDPOINT", "https://api.textsynth.com/v1/engines/{engine}/completions")
    _API_KEY = os.environ.get('TEXTSYNTH_API_SECRET_KEY', None)
    _NAME = "textsynth"

    def __init__(self, model="QA"):
        self._model = model
        assert self._API_KEY is not None, "Please set FOREFRONT_API_KEY env var for running through Forefront"
		
    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "max_tokens": genparams.max_tokens,
            "temperature":genparams.temperature,
            "stop":genparams.stop,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result['completion'] = result['raw']['text']
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = completion['raw']['input_tokens']
        tk2 = completion['raw']['output_tokens']
        return tk1, tk2

class AnthropicModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/complete")
    _API_KEY = os.environ.get('ANTHROPIC_API_KEY', None)
    _NAME = "anthropic"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set ANTHROPIC_API_KEY env var for running through anthropic"
        #self.client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])
        self.client = anthropic.Anthropic()
        #self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            #"echo": False,
            "max_tokens_to_sample": genparams.max_tokens,
            #"logprobs": 1,
            "temperature": genparams.temperature,
            #"top_p": 1,
            #"stop":, # seems it does not allow \n to be the stopping token.
            "model":self._model,
            #"role":"user",
        }
        #print("the request is---")
        #print(req)
        return req
    
    def _response_format(self,
                         response,
                         ):
        #print("response is",response)
        result = dict()
        result['raw'] = response
        result["completion"] = response.content[0].text
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        usage = completion['raw'].usage
        tk1 = usage.input_tokens
        tk2 = usage.output_tokens
        #print(completion['raw'].usage.input_tokens)
        #tk1 = anthropic.count_tokens(context)
        #tk2 = anthropic.count_tokens(completion['completion'])
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        '''
        response = self.client.completion(
            prompt=anthropic.HUMAN_PROMPT+data['prompt']+anthropic.AI_PROMPT,
            model=data['model'],
            max_tokens_to_sample=data['max_tokens_to_sample'],
            )  
        '''
        while(1):
            try:
                response = self.client.messages.create(
            model=data['model'],
            max_tokens=data['max_tokens_to_sample'],
            temperature=data['temperature'],
            #stop_sequences=['\n\n'],
            system="Follow the examples to only generate the answer. Do not generate any other texts.",

            messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{data['prompt']}"
                    }
                ],
            }
        ],
            )
                return response      
            except:
              time.sleep(retry_grace_time)

class GoogleModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://api.anthropic.com/v1/complete")
    _API_KEY = os.environ.get('GEMINI_API_KEY', None)
    _NAME = "google"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set GEMINI_API_KEY env var for running through gemini models"

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            #"echo": False,
            "max_tokens_to_sample": genparams.max_tokens,
            #"logprobs": 1,
            "temperature": genparams.temperature,
            #"top_p": 1,
            #"stop":, # seems it does not allow \n to be the stopping token.
            "model":self._model,
            #"role":"user",
        }
        #print("the request is---")
        #print(req)
        return req
    
    def _response_format(self,
                         response,
                         ):
        #print("response is",response)
        result = dict()
        result['raw'] = response
        result["completion"] = response.candidates[0].content.parts[0].text
        return result    
    
    def _get_io_tokens(self,context, completion):
        usage = completion['raw'].usage_metadata
        tk1 = usage.prompt_token_count
        tk2 = usage.candidates_token_count
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        while(1):
            try:
                generation_config = {
        "temperature": data['temperature'],
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": data['max_tokens_to_sample'],
        "response_mime_type": "text/plain",
        }
                model = genai.GenerativeModel(
        model_name=data['model'],
        generation_config=generation_config,
        )
                chat_session = model.start_chat(
          history=[
            ]
        )

                response = chat_session.send_message(data['prompt'])

                return response 
            except Exception as e:
                print(f"Failed with errors {e}, retry")
                time.sleep(retry_grace_time)

class TogetherAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("TOGETHERAI_ENDPOINT", "https://api.together.ai/v1/chat/completions")
    _API_KEY = os.environ.get('TOGETHER_API_KEY', None)
    _NAME = "togetherai"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set TOGETHER_API_KEY env var for running through Together AI"
    
    def _request_format(self, context, genparams):
        req = {
            "model": self._model,
            "messages": [{"content": context, "role": "user"}],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 0.7,  # Defaulting to values from your example code
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": genparams.stop,
            "stream": False,
        }
        return req

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response['choices'][0]['message']['content']
        return result

    def _get_io_tokens(self, context, completion):
        # If Together AI has token usage, adapt this accordingly
        tk1 = completion['raw'].get('usage', {}).get('prompt_tokens', 0)
        tk2 = completion['raw'].get('usage', {}).get('completion_tokens', 0)
        return tk1, tk2
    

class AzureLlamaModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "https://your-llama-model-endpoint")
    _API_KEY = os.environ.get("LLAMA_API_KEY", None)
    _NAME = "azure_llama"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AZURE_INFERENCE_CREDENTIAL env var for running through Azure AI"
        self.client = ChatCompletionsClient(
            endpoint=self._ENDPOINT,
            credential=AzureKeyCredential(self._API_KEY)
        )

    def _request_format(self, context, genparams):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 1.0,
            "stop": genparams.stop
        }
        return payload

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        tk1 = completion['raw'].usage.prompt_tokens
        tk2 = completion['raw'].usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.complete(data)
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")
    



class AzureCohereModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("COHERE_STUDIO_ENDPOINT", "https://cohere-command-r-fzcrf.eastus.models.ai.azure.com")
    _API_KEY = os.environ.get("COHERE_STUDIO_API_KEY", None)
    _NAME = "azure_cohere"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AZURE_INFERENCE_CREDENTIAL env var for running through Azure AI"
        self.client = ChatCompletionsClient(
            endpoint=self._ENDPOINT,
            credential=AzureKeyCredential(self._API_KEY)
        )

    def _request_format(self, context, genparams):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": genparams.stop
        }
        return payload

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        tk1 = completion['raw'].usage.prompt_tokens
        tk2 = completion['raw'].usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.complete(data)
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")
    

class AzureMistralNemoModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("MISTRAL_NEMO_ENDPOINT", "https://Mistral-Nemo-maamt.eastus.models.ai.azure.com")
    _API_KEY = os.environ.get("MISTRAL_NEMO_API_KEY", None)
    _NAME = "azure_mistral_nemo"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AZURE_INFERENCE_CREDENTIAL env var for running through Azure AI"
        self.client = ChatCompletionsClient(
            endpoint=self._ENDPOINT,
            credential=AzureKeyCredential(self._API_KEY)
        )

    def _request_format(self, context, genparams):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 1.0,
            "stop": genparams.stop
        }
        return payload

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        tk1 = completion['raw'].usage.prompt_tokens
        tk2 = completion['raw'].usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.complete(data)
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")



class AzureMinistralModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("MINISTRAL_ENDPOINT", "https://Ministral-3B-topie.eastus.models.ai.azure.com")
    _API_KEY = os.environ.get("MINISTRAL_API_KEY", None)
    _NAME = "azure_ministral"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AZURE_INFERENCE_CREDENTIAL env var for running through Azure AI"
        logging.info(f"Connecting to endpoint: {self._ENDPOINT}")
        self.client = ChatCompletionsClient(
            endpoint=self._ENDPOINT,
            credential=AzureKeyCredential(self._API_KEY)
        )

    def _request_format(self, context, genparams):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": genparams.stop
        }
        return payload

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        tk1 = completion['raw'].usage.prompt_tokens
        tk2 = completion['raw'].usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.complete(data)
                return response
            except Exception as e:
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")
    

class AzureOpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_GPT35_ENDPOINT", "https://cognitiveservices.openai.azure.com/")
    _DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat")
    _API_KEY = os.environ.get("OPENAI_GPT35_API_KEY", None)
    _NAME = "azure_openai"
    
    def __init__(self, model):
        self._model = model
        self.token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=self._ENDPOINT,
            azure_ad_token_provider=self.token_provider,
            api_version="2024-05-01-preview",
        )
        
    
    def _request_format(self, context, genparams):
        chat_prompt = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role": "user",
                "content": context
            }
        ]

        return chat_prompt

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._DEPLOYMENT_NAME,
                    messages=data,
                    max_tokens=2000, #self.genparams.max_tokens
                    temperature=0.7, #self.genparams.max_tokens
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False
                )
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")
    
    def _get_io_tokens(self, context, completion):
        usage = completion['raw'].usage
        tk1 = usage.prompt_tokens
        tk2 = usage.completion_tokens
        return tk1, tk2
    

class AzureGPT4oModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_4o_ENDPOINT", "https://azure-openai-dev-001.openai.azure.com/")
    _DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "gpt-4o")
    _API_KEY = os.environ.get("OPENAI_4o_API_KEY", None)
    _NAME = "azure_gpt4o"

    def __init__(self, model):
        self._model = model
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self._ENDPOINT,
            api_key=self._API_KEY,
        )

    def _request_format(self, context, genparams):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            },
            {
                "role": "user",
                "content": context
            }
        ]
        return chat_prompt

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        usage = completion['raw'].usage
        tk1 = usage.prompt_tokens
        tk2 = usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._DEPLOYMENT_NAME,
                    messages=data,
                    max_tokens=2000,  # self.genparams.max_tokens
                    temperature=0.7,  # self.genparams.temperature
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False
                )
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")
    
    def request_format(self, query, context, system_prompt, few_shots):
        chat_prompt = []
        if query :
            chat_prompt.append({"role": "user", "content": query})

        if system_prompt:
            chat_prompt.append({
                "role": "system",
                "content": system_prompt
            })
        
        for shot in few_shots:
            chat_prompt.append({"role": "user", "content": shot["content"]})
            chat_prompt.append({"role": "assistant", "content": shot["content"]})
        
        chat_prompt.append({"role": "user", "content": context})
        return chat_prompt

    def response_format(self, response):
        return {
            "raw": response,
            "completion": response.choices[0].message.content
        }

    def getcompletiongpt4o(self, query, genparams, content, system_prompt="", few_shots=[]):
        messages = self.request_format(query, content, system_prompt, few_shots)
        
        try:
            response = self.client.chat.completions.create(
                model=self._DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=genparams.max_tokens,
                temperature=genparams.temperature,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=genparams.stop,
                stream=False
            )
            return self.response_format(response)["completion"]
        except Exception as e:
            print(f"Error during API call: {e}")
            return None


class AzureGPT4oMiniModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_4omini_ENDPOINT", "https://azure-openai-dev-001.openai.azure.com/")
    _DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "gpt-4o-mini")
    _API_KEY = os.environ.get("OPENAI_4omini_API_KEY", None)
    _NAME = "azure_gpt4o_mini"

    def __init__(self, model):
        self._model = model
        self.token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=self._ENDPOINT,
            azure_ad_token_provider=self.token_provider,
            api_version="2024-05-01-preview",
        )

    def _request_format(self, context, genparams):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            },
            {
                "role": "user",
                "content": context
            }
        ]
        return chat_prompt

    def _response_format(self, response):
        result = dict()
        result['raw'] = response
        result["completion"] = response.choices[0].message.content
        return result

    def _get_io_tokens(self, context, completion):
        usage = completion['raw'].usage
        tk1 = usage.prompt_tokens
        tk2 = usage.completion_tokens
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for _ in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._DEPLOYMENT_NAME,
                    messages=data,
                    max_tokens=2000,  # self.genparams.max_tokens
                    temperature=0.7,  # self.genparams.temperature
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False
                )
                return response
            except Exception as e:
                print(f"Failed to call API: {e}, retrying...")
                time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")



_PROVIDER_MAP = {
    #"openai": OpenAIModelProvider, # cleaned
    #"ai21": AI21ModelProvider, # cleaned
    #"cohere":CohereAIModelProvider, # cleaned
    #"openaichat":OpenAIChatModelProvider,# cleaned
    #"llama": LlamaModelProvider,  # Ajout de LlamaModelProvider
    #"azure_ai21": AI21ModelProvider,
    #"azure_cohere": AzureCohereModelProvider,
    
    "azure_openai": AzureOpenAIModelProvider,
    "azure_llama": AzureLlamaModelProvider,
    "azure_mistral_nemo" : AzureMistralNemoModelProvider,
    "azure_ministral" : AzureMinistralModelProvider,
    "azure_gpt4o": AzureGPT4oModelProvider,
    "azure_gpt4o_mini": AzureGPT4oMiniModelProvider,

}



#"anthropic":AnthropicModelProvider,
#"togetherai":TogetherAIModelProvider,
#"google":GoogleModelProvider,
#"forefrontai":ForeFrontAIModelProvider, # cleaned
#"textsynth":TextSynthModelProvider, # cleaned

def make_model(provider, model):
    assert provider in _PROVIDER_MAP, f"No model provider '{provider}' implemented"
    return _PROVIDER_MAP[provider](model)
