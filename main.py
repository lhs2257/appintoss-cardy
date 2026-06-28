import os
import re
import json
import base64
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageOps
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

PROMPT = """이 명함 이미지에서 정보를 추출해서 아래 JSON 형식으로만 응답해주세요. 해당 정보가 없으면 null로 응답하세요.

{"name":null,"company":null,"role":null,"phone":null,"email":null}

규칙:
- name: 사람 이름만 (직책어·회사명 제외), 공백 없이 붙여쓰기 (예: "이준섭")
- company: 회사·기관 전체 이름 (예: "KAP 한국자산매입")
- role: 부서명 포함 직책 (예: "상품R&D센터 AI LAB Leader", "마케팅팀 팀장")
- phone: 010-XXXX-XXXX 형식으로 정규화, FAX 번호 제외, 없으면 null
- email: 이메일 주소만, 없으면 null"""


class ScanRequest(BaseModel):
    image: str
    rotation: int = 0
    portrait: bool = False
    cropRect: dict | None = None  # legacy
    corners: list | None = None   # [{x,y}×4] 정규화 0-1, 회전 후 좌표계


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def warp_with_corners(img: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """사용자 지정 4 꼭짓점으로 1차 warp 후, 결과 이미지에서 카드 경계 재인식해 2차 warp."""
    if len(pts) != 4:
        return None
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    maxW = min(int(max(wA, wB)), 1400)
    maxH = min(int(max(hA, hB)), 1400)
    if maxW < 20 or maxH < 20:
        print(f"[corners] output too small {maxW}x{maxH}", flush=True)
        return None
    print(f"[corners] 1st warp {maxW}x{maxH}", flush=True)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    pre_corrected = cv2.warpPerspective(img, M, (maxW, maxH))

    # 2차: 1차 결과에서 카드 경계를 자동 인식해 배경 제거
    refined = perspective_correct(pre_corrected)
    if refined is not None:
        print(f"[corners] 2nd pass ok {refined.shape[1]}x{refined.shape[0]}", flush=True)
        return refined
    print("[corners] 2nd pass not found, using 1st warp", flush=True)
    return pre_corrected


def perspective_correct(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    threshold = w * h * 0.05
    print(f"[persp] img={w}x{h}, contours={len(contours)} thr={threshold:.0f}", flush=True)

    for i, c in enumerate(contours):
        raw_area = cv2.contourArea(c)
        if raw_area < threshold:
            print(f"[persp] c#{i} area={raw_area:.0f} below threshold, stop", flush=True)
            break

        peri = cv2.arcLength(c, True)
        approx = None
        for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
            a = cv2.approxPolyDP(c, eps * peri, True)
            if len(a) == 4:
                approx = a
                break
            print(f"[persp] c#{i} area={raw_area:.0f} eps={eps} pts={len(a)}", flush=True)

        if approx is None:
            print(f"[persp] c#{i} no 4-pt approx, skip", flush=True)
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        print(f"[persp] c#{i} approxPolyDP ok", flush=True)

        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        wA = float(np.linalg.norm(br - bl))
        wB = float(np.linalg.norm(tr - tl))
        hA = float(np.linalg.norm(tr - br))
        hB = float(np.linalg.norm(tl - bl))
        maxW = min(int(max(wA, wB)), 1200)
        maxH = min(int(max(hA, hB)), 1200)
        if maxW < 100 or maxH < 60:
            print(f"[persp] c#{i} too small {maxW}x{maxH}", flush=True)
            continue
        print(f"[persp] SUCCESS c#{i} {maxW}x{maxH}", flush=True)
        dst = np.array(
            [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxW, maxH))

    print("[persp] FAIL: no valid card", flush=True)
    return None


def enhance_image(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    pil_img = ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(1.25)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


_ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def decode_image(img_data: bytes) -> np.ndarray:
    """EXIF 방향 보정 후 OpenCV 배열로 변환. Pillow 실패 시 cv2 폴백."""
    print(f"[decode] bytes={len(img_data)} header={img_data[:8].hex()}", flush=True)
    try:
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_data))).convert("RGB")
        print(f"[decode] pillow ok size={pil_img.size} mode={pil_img.mode}", flush=True)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[decode] pillow fail: {e}, trying cv2", flush=True)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지 디코딩 실패")
        print(f"[decode] cv2 ok shape={img.shape}", flush=True)
        return img


def fix_orientation(img: np.ndarray, portrait: bool) -> np.ndarray:
    """앱에서 선택한 크롭 방향과 보정 결과 방향이 불일치하면 90° 회전."""
    h, w = img.shape[:2]
    is_landscape = w > h
    if portrait and is_landscape:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if not portrait and not is_landscape:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def process_image(
    base64_str: str,
    rotation: int = 0,
    portrait: bool = False,
    crop_rect: dict | None = None,
    corners: list | None = None,
) -> str:
    print(f"[proc] b64_len={len(base64_str)} rotation={rotation} corners={bool(corners)} crop={bool(crop_rect)}", flush=True)
    img_data = base64.b64decode(base64_str)

    img = decode_image(img_data)
    max_dim = max(img.shape[:2])
    if max_dim > 1600:
        scale = 1600 / max_dim
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"[proc] resized to {img.shape[1]}x{img.shape[0]}", flush=True)

    if rotation in _ROTATE_MAP:
        img = cv2.rotate(img, _ROTATE_MAP[rotation])
        print(f"[proc] rotated {rotation}deg → {img.shape[1]}x{img.shape[0]}", flush=True)

    if corners:
        # 사용자 지정 4 꼭짓점 → 직접 warpPerspective
        h2, w2 = img.shape[:2]
        pts = np.array([[c['x'] * w2, c['y'] * h2] for c in corners], dtype=np.float32)
        corrected = warp_with_corners(img, pts)
        if corrected is not None:
            enhanced = enhance_image(corrected)
        else:
            enhanced = enhance_image(img)
    elif crop_rect:
        # legacy: 직사각형 크롭 후 자동 perspective
        h2, w2 = img.shape[:2]
        x1 = max(0, int(crop_rect['x'] * w2))
        y1 = max(0, int(crop_rect['y'] * h2))
        x2 = min(w2, int((crop_rect['x'] + crop_rect['w']) * w2))
        y2 = min(h2, int((crop_rect['y'] + crop_rect['h']) * h2))
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]
        corrected = perspective_correct(img)
        if corrected is not None:
            gray_mean = float(cv2.mean(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))[0])
            if gray_mean < 40:
                corrected = None
        if corrected is not None:
            corrected = fix_orientation(corrected, portrait)
            enhanced = enhance_image(corrected)
        else:
            enhanced = enhance_image(img)
    else:
        corrected = perspective_correct(img)
        if corrected is not None:
            corrected = fix_orientation(corrected, portrait)
            enhanced = enhance_image(corrected)
        else:
            enhanced = enhance_image(img)

    _, buffer = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[proc] done encoded={len(buffer)}bytes", flush=True)
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(req: ScanRequest):
    raw_base64 = req.image
    if "base64," in raw_base64:
        raw_base64 = raw_base64.split("base64,", 1)[1]

    processed_base64: str | None = None
    try:
        processed_base64 = process_image(raw_base64, req.rotation, req.portrait, req.cropRect, req.corners)
    except Exception:
        # 처리 실패 시 원본 이미지를 GPT에 전달하되 correctedImage는 반환하지 않음
        pass

    ocr_image = processed_base64 if processed_base64 is not None else raw_base64

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ocr_image}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            OPENAI_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json=payload,
        )

    if not resp.is_success:
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="API 키 오류")
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="요청 한도 초과, 잠시 후 다시 시도해주세요")
        raise HTTPException(status_code=502, detail=f"OpenAI 오류: {resp.status_code}")

    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"]

    json_match = re.search(r'\{[\s\S]*\}', raw_text)
    if not json_match:
        raise HTTPException(status_code=500, detail="응답 파싱 실패")

    parsed = json.loads(json_match.group())

    return {
        "name": parsed.get("name"),
        "company": parsed.get("company"),
        "role": parsed.get("role"),
        "phone": parsed.get("phone"),
        "email": parsed.get("email"),
        "rawText": raw_text,
        "correctedImage": processed_base64,  # 처리 실패 시 None → 앱이 원본 cropRect 표시
    }
