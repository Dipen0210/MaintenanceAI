"""Download official, revision-pinned safetensors models; record every local file hash."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'artifacts/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')


def main():
    from huggingface_hub import HfApi,snapshot_download
    manifest_path=ROOT/'models/language-models.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries={}
    for key,repo in [('embedding','sentence-transformers/all-MiniLM-L6-v2'),('language','Qwen/Qwen3-0.6B')]:
        revision=previous.get(key,{}).get('revision') or HfApi(token=False).model_info(repo).sha
        directory=ROOT/'artifacts/language_models'/key
        snapshot_download(repo_id=repo,revision=revision,local_dir=directory,token=False,
                          allow_patterns=['*.json','*.safetensors','vocab.txt','vocab.json','merges.txt','tokenizer.model','1_Pooling/config.json','LICENSE','README.md'],
                          ignore_patterns=['onnx/*','openvino/*'],max_workers=2)
        files={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file() and '.cache' not in p.parts}
        entries[key]={'repository':repo,'revision':revision,'license':'Apache-2.0','source':'https://huggingface.co/'+repo,
                      'directory':str(directory.relative_to(ROOT)),'files':files}
        print(key,revision,len(files),'files verified',flush=True)
    manifest_path.write_text(json.dumps(entries,indent=2)+'\n')

if __name__=='__main__':main()
