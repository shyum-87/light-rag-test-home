# BAAI/bge-m3 ONNX 모델 파일 위치

이 폴더에 ONNX 변환된 bge-m3 모델 파일을 넣어주세요.
torch 없이 onnxruntime만으로 임베딩이 가능합니다.

## 변환 방법 (인터넷 되는 PC에서)

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model BAAI/bge-m3 ./models/bge-m3-onnx
```

## 변환 후 폴더 구조

```
models/bge-m3-onnx/
  config.json              (1KB)
  model.onnx               (424KB - ONNX 그래프)
  model.onnx_data          (2.2GB - 모델 가중치)
  tokenizer.json           (17MB)
  tokenizer_config.json    (1KB)
  special_tokens_map.json  (1KB)
  sentencepiece.bpe.model  (5MB)
```

## 폐쇄망 전달

변환 완료 후 이 폴더째 USB 등으로 폐쇄망 PC에 복사하면 됩니다.
총 크기: 약 2.3GB
