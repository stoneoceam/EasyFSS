import argparse
import base64
import json
import mimetypes
import os
import shutil
import sys
import time
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
RUNTIME_ROOT = DEMO_ROOT / "runtime"
UPLOAD_ROOT = RUNTIME_ROOT / "uploads"
OUTPUT_ROOT = RUNTIME_ROOT / "outputs"
LOGS_ROOT = PROJECT_ROOT / "logs"

def clear_runtime_root(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


for path in (UPLOAD_ROOT, OUTPUT_ROOT):
    clear_runtime_root(path)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_DINO_SIZES = {"small", "base", "large"}
VALID_SAM2_SIZES = {"small", "base", "large"}
SIZE_SHORT = {
    "small": "s",
    "base": "b",
    "large": "l",
}


@dataclass
class UploadedFile:
    filename: str
    content_type: str
    data: bytes


def normalize_request_id(request_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request_id)
    return cleaned[:80] or f"easyfss-{int(time.time())}"


def build_model_variant_name(dinov2_size: str, sam2_size: str) -> str:
    if dinov2_size == "base" and sam2_size == "base":
        return "net"
    return f"net_{SIZE_SHORT[sam2_size]}{SIZE_SHORT[dinov2_size]}"


def resolve_ckpt_path(dinov2_size: str, sam2_size: str) -> Path:
    ckpt_override = os.environ.get("EASYFSS_DEMO_CKPT")
    if ckpt_override:
        ckpt_path = Path(ckpt_override).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到指定权重文件: {ckpt_path}")
        return ckpt_path

    benchmark = os.environ.get("EASYFSS_DEMO_BENCHMARK", "coco")
    fold = os.environ.get("EASYFSS_DEMO_FOLD", "0")
    variant_name = build_model_variant_name(dinov2_size, sam2_size)
    ckpt_path = LOGS_ROOT / variant_name / f"{benchmark}-{fold}.log" / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            "找不到权重文件: "
            f"{ckpt_path}。可通过 EASYFSS_DEMO_CKPT 指定自定义 best.pt。"
        )

    return ckpt_path


def upload_suffix(upload_file: UploadedFile, fallback: str) -> str:
    return Path(upload_file.filename or "").suffix or fallback


def save_uploaded_file(upload_file: UploadedFile, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(upload_file.data)


def image_data_url(path: str) -> str:
    file_path = Path(path)
    content_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
    data = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{data}"


def remove_runtime_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def parse_multipart_form(content_type: str, body: bytes):
    if "multipart/form-data" not in content_type:
        raise ValueError("请求必须是 multipart/form-data")

    raw_message = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=default).parsebytes(raw_message)

    fields = {}
    files = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition")
        if not disposition:
            continue

        name = part.get_param("name", header="content-disposition")
        if not name:
            continue

        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is None:
            charset = part.get_content_charset() or "utf-8"
            fields.setdefault(name, []).append(payload.decode(charset, errors="replace"))
        else:
            files.setdefault(name, []).append(
                UploadedFile(
                    filename=Path(filename).name,
                    content_type=part.get_content_type() or "application/octet-stream",
                    data=payload,
                )
            )

    return fields, files


def first_field(fields, name: str, default=None):
    values = fields.get(name)
    if not values:
        return default
    return values[0]


def run_prediction(fields, files):
    from demo.demo_infer import run_demo_inference_multi

    start_time = time.time()
    request_id = normalize_request_id(first_field(fields, "request_id", ""))
    dinov2_size = first_field(fields, "dinov2_size", "base")
    sam2_size = first_field(fields, "sam2_size", "base")

    if dinov2_size not in VALID_DINO_SIZES:
        return {"status": "failed", "error": {"message": f"不支持的 dinov2_size: {dinov2_size}"}}
    if sam2_size not in VALID_SAM2_SIZES:
        return {"status": "failed", "error": {"message": f"不支持的 sam2_size: {sam2_size}"}}

    support_images = files.get("support_images", [])
    support_masks = files.get("support_masks", [])
    query_images = files.get("query_images", []) or files.get("query_image", [])

    if len(support_images) != len(support_masks):
        return {"status": "failed", "error": {"message": "support_images 和 support_masks 数量不一致"}}
    if not support_images:
        return {"status": "failed", "error": {"message": "support_images 不能为空"}}
    if not query_images:
        return {"status": "failed", "error": {"message": "query_images 不能为空"}}

    ckpt_path = resolve_ckpt_path(dinov2_size, sam2_size)
    variant_name = build_model_variant_name(dinov2_size, sam2_size)

    upload_dir = UPLOAD_ROOT / request_id
    output_dir = OUTPUT_ROOT / request_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_img_paths = []
    support_mask_paths = []
    for index, (img, mask) in enumerate(zip(support_images, support_masks)):
        img_path = upload_dir / f"support_img_{index}{upload_suffix(img, '.jpg')}"
        mask_path = upload_dir / f"support_mask_{index}{upload_suffix(mask, '.png')}"
        save_uploaded_file(img, img_path)
        save_uploaded_file(mask, mask_path)
        support_img_paths.append(str(img_path))
        support_mask_paths.append(str(mask_path))

    query_img_paths = []
    for index, query_image in enumerate(query_images):
        query_img_path = upload_dir / f"query_{index}{upload_suffix(query_image, '.jpg')}"
        save_uploaded_file(query_image, query_img_path)
        query_img_paths.append(str(query_img_path))

    query_mask_paths = [""] * len(query_images)

    pred_paths = [
        str(output_dir / f"pred_mask_{index}.png")
        for index in range(len(query_images))
    ]

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        result = run_demo_inference_multi(
            support_img_paths=support_img_paths,
            support_mask_paths=support_mask_paths,
            query_img_paths=query_img_paths,
            query_mask_paths=query_mask_paths,
            ckpt_path=str(ckpt_path),
            save_pred_paths=pred_paths,
            sam2_backbone_size=sam2_size,
            dinov2_backbone_size=dinov2_size,
            size=420,
            class_id=0,
            use_amp=True,
            amp_dtype="bfloat16",
            normalize=False,
        )
        items = []
        for index, item in enumerate(result.get("items", [])):
            pred_mask_url = image_data_url(pred_paths[index])
            items.append({
                **item,
                "pred_mask_url": pred_mask_url,
            })

        pred_mask_urls = [item["pred_mask_url"] for item in items]

        return {
            "status": "success",
            "outputs": {
                "pred_mask_url": pred_mask_urls[0] if pred_mask_urls else "",
                "pred_mask_urls": pred_mask_urls,
                "items": items,
            },
            "meta": {
                "time_cost": round(time.time() - start_time, 3),
                "query_count": result.get("query_count", len(query_images)),
                "batch_count": result.get("batch_count"),
                "max_query_batch": result.get("max_query_batch"),
                "variant_name": variant_name,
                "dinov2_size": dinov2_size,
                "sam2_size": sam2_size,
                "ckpt_path": str(ckpt_path),
            },
        }
    finally:
        os.chdir(old_cwd)
        remove_runtime_dir(upload_dir)
        remove_runtime_dir(output_dir)


def safe_join(base: Path, request_path: str) -> Path:
    target = (base / request_path.lstrip("/")).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError("非法路径")
    return target


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "EasyFSSDemo/1.0"

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.add_default_headers()
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path in {"/", "/index.html"}:
            self.send_file(DEMO_ROOT / "index.html")
            return

        if path == "/health":
            self.send_json({"status": "easyfss demo ok"})
            return

        if path.startswith("/css/"):
            self.send_safe_file(DEMO_ROOT, path)
            return

        if path.startswith("/js/"):
            self.send_safe_file(DEMO_ROOT, path)
            return

        if path.startswith("/outputs/"):
            self.send_safe_file(OUTPUT_ROOT, path.removeprefix("/outputs/"))
            return

        if path.startswith("/api/easyfss/outputs/"):
            self.send_safe_file(OUTPUT_ROOT, path.removeprefix("/api/easyfss/outputs/"))
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path != "/api/easyfss/predict":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            fields, files = parse_multipart_form(self.headers.get("Content-Type", ""), body)
            result = run_prediction(fields, files)
            self.send_json(result)
        except Exception as exc:
            self.send_json(
                {"status": "failed", "error": {"message": str(exc)}},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def send_safe_file(self, base: Path, request_path: str):
        try:
            self.send_file(safe_join(base, request_path))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def send_file(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.add_default_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

        with file_path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.add_default_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message: str):
        self.send_json({"status": "failed", "error": {"message": message}}, status=status)

    def add_default_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description="EasyFSS demo server without FastAPI")
    parser.add_argument("--host", default=os.environ.get("EASYFSS_DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EASYFSS_DEMO_PORT", "9000")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"EasyFSS demo running at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
