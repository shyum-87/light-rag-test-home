# BAAI/bge-m3 모델 파일 위치

이 폴더에 HuggingFace에서 다운로드한 BAAI/bge-m3 모델 파일을 넣어주세요.

## 다운로드 방법 (인터넷 되는 PC에서)

### 방법 1: huggingface-cli (권장)

```bash
pip install huggingface_hub
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
```

### 방법 2: Python 스크립트

```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3", local_dir="./models/bge-m3")
```

### 방법 3: git clone

```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-m3 ./models/bge-m3
```

## 다운로드 후 예상 폴더 구조

```
models/bge-m3/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  sentencepiece.bpe.model
  model.safetensors        (또는 pytorch_model.bin)
  colbert_linear.pt
  sparse_linear.pt
  ...
```

## 폐쇄망 전달

다운로드 완료 후 이 폴더째 USB 등으로 폐쇄망 PC에 복사하면 됩니다.
모델 전체 크기: 약 2.3GB
