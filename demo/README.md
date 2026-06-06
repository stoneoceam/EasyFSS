# EasyFSS Demo

This directory contains the local demo page ported from the original Demo_System frontend. The model runtime uses the `EasyFSS` environment. The server entry point is implemented with the Python standard library and does not depend on FastAPI, uvicorn, or python-multipart.

## Launch

```bash
conda activate EasyFSS
bash demo/start_demo.sh
```

Open `http://127.0.0.1:9000/` to access the demo page. The page supports K support image/mask pairs and multiple query images. Query images are inferred in batches of up to 20 images; larger requests are split automatically.

## Default Checkpoint

By default, the backend loads the checkpoint from `logs/<variant>/coco-0.log/best.pt` in the current project, where `<variant>` is automatically matched according to the SAM2 and DINOv2 sizes selected on the page. You can also specify the checkpoint and evaluation settings with environment variables:

```bash
EASYFSS_DEMO_CKPT=/path/to/best.pt bash demo/start_demo.sh
EASYFSS_DEMO_BENCHMARK=pascal EASYFSS_DEMO_FOLD=1 bash demo/start_demo.sh
```

Uploaded files, caches, and prediction results are written to `demo/runtime/`.
