$ docker login nvcr.io
Username: $oauthtoken
Password: <PASTE_API_KEY_HERE>


export NGC_API_KEY=<your key>

docker run -it --rm --name=parakeet-1-1b-rnnt-multilingual \
   --gpus all \
   --shm-size=8GB \
   -e NGC_API_KEY \
   -e NIM_HTTP_API_PORT=9000 \
   -e NIM_GRPC_API_PORT=50051 \
   -p 9001:9000 \
   -p 50052:50051 \
   -e NIM_TAGS_SELECTOR=mode=str \
   nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual:latest




## TTS Magpie multilingual

```
docker run -it --rm --name=magpie-tts-multilingual \
    --gpus all \
    --shm-size=8GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -e NIM_HTTP_API_PORT=9000 \
    -e NIM_GRPC_API_PORT=50051 \
    -p 9000:9000 \
    -p 50051:50051 \
    nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
```