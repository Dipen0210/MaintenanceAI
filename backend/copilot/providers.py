"""Provider-neutral completion contract with verified local inference and explicit failures."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib,json,os,threading,time
from typing import Protocol
from backend.core.database import ROOT


@dataclass(frozen=True)
class Completion:
    text:str
    model_id:str
    input_tokens:int
    output_tokens:int
    latency_ms:float
    estimated_cost_usd:float|None
    cost_basis:str


class LanguageProvider(Protocol):
    def complete(self,messages:list[dict],max_tokens:int=256,schema:dict|None=None)->Completion:...


@lru_cache(maxsize=2)
def verified_directory(kind):
    manifest=json.loads((ROOT/'models/language-models.json').read_text())[kind]
    directory=(ROOT/manifest['directory']).resolve()
    if ROOT/'artifacts' not in directory.parents:raise ValueError('Language model path is outside artifact storage')
    for filename,digest in manifest['files'].items():
        path=directory/filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Language model artifact is missing or corrupt')
    return directory,manifest


class LocalTransformersProvider:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        directory,manifest=verified_directory('language')
        torch.set_num_threads(int(os.getenv('MAINTENANCEAI_CPU_THREADS','2')))
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
                                                      dtype=torch.float32).eval()
        self.model_id=manifest['repository']+'@'+manifest['revision'];self.lock=threading.Lock()

    def complete(self,messages,max_tokens=256,schema=None):
        import torch
        started=time.perf_counter()
        with self.lock:
            prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=self.tokenizer(prompt,return_tensors='pt')
            count=inputs['input_ids'].shape[-1]
            if count>6000:raise ValueError('Language-model context exceeds the configured evidence budget')
            constrained={}
            if schema:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
                constrained['prefix_allowed_tokens_fn']=build_transformers_prefix_allowed_tokens_fn(self.tokenizer,JsonSchemaParser(schema))
            with torch.inference_mode():
                generated=self.model.generate(**inputs,max_new_tokens=min(max_tokens,512),do_sample=False,
                                              pad_token_id=self.tokenizer.eos_token_id,**constrained)
            output=generated[0,count:]
            text=self.tokenizer.decode(output,skip_special_tokens=True).strip()
            return Completion(text,self.model_id,int(count),len(output),round((time.perf_counter()-started)*1000,3),None,
                              'Local CPU inference; no metered API charge. Electricity/hardware cost is not estimated.')


class HTTPChatProvider:
    """Explicit administrator configuration only; users cannot choose an endpoint or secret."""
    def __init__(self,url,model,api_key=None):
        from urllib.parse import urlsplit
        parsed=urlsplit(url)
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ['localhost','127.0.0.1']):
            raise ValueError('Remote providers require HTTPS; local providers may use loopback HTTP')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Provider URL must not contain credentials or query parameters')
        self.url=url;self.model=model;self.api_key=api_key

    def complete(self,messages,max_tokens=256,schema=None):
        import httpx
        started=time.perf_counter();headers={'Authorization':'Bearer '+self.api_key} if self.api_key else {}
        payload={'model':self.model,'messages':messages,'max_tokens':min(max_tokens,512),'temperature':0}
        if schema:payload['response_format']={'type':'json_schema','json_schema':{'name':'evidence_selection','strict':True,'schema':schema}}
        response=httpx.post(self.url,json=payload,
                            headers=headers,timeout=90,follow_redirects=False)
        response.raise_for_status();payload=response.json();usage=payload['usage']
        return Completion(payload['choices'][0]['message']['content'],str(payload.get('model',self.model)),
                          int(usage['prompt_tokens']),int(usage['completion_tokens']),round((time.perf_counter()-started)*1000,3),None,
                          'Provider usage reported; pricing is not configured, so cost is unknown.')


@lru_cache(maxsize=1)
def _language_provider():
    provider=os.getenv('MAINTENANCEAI_LLM_PROVIDER','local')
    if provider=='local':return LocalTransformersProvider()
    if provider=='http':return HTTPChatProvider(os.environ['MAINTENANCEAI_LLM_URL'],os.environ['MAINTENANCEAI_LLM_MODEL'],os.getenv('MAINTENANCEAI_LLM_API_KEY'))
    raise ValueError('Unsupported language provider')


_provider_initialization_lock=threading.Lock()
def language_provider():
    with _provider_initialization_lock:return _language_provider()


@lru_cache(maxsize=1)
def embedding_model():
    from transformers import AutoTokenizer,AutoModel
    directory,manifest=verified_directory('embedding')
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(directory,local_files_only=True,trust_remote_code=False).eval()
    return tokenizer,model,manifest['repository']+'@'+manifest['revision']


def embed(texts):
    import torch
    import torch.nn.functional as F
    from backend.copilot.configuration import RETRIEVAL
    tokenizer,model,version=embedding_model()
    encoded=tokenizer(texts,padding=True,truncation=True,max_length=RETRIEVAL['embedding_tokens'],return_tensors='pt')
    with torch.inference_mode():
        output=model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        pooled=(output*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        vectors=F.normalize(pooled,p=2,dim=1).cpu().numpy()
    return vectors,version
