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
from PIL import Image, ImageEnhance

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


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def perspective_correct(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < w * h * 0.1:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        wA = float(np.linalg.norm(br - bl))
        wB = float(np.linalg.norm(tr - tl))
        hA = float(np.linalg.norm(tr - br))
        hB = float(np.linalg.norm(tl - bl))
        maxW = min(int(max(wA, wB)), 1200)
        maxH = min(int(max(hA, hB)), 800)
        if maxW < 100 or maxH < 60:
            continue
        dst = np.array(
            [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxW, maxH))

    return None


def enhance_image(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    pil_img = ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(1.25)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_image(base64_str: str) -> str:
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    corrected = perspective_correct(img)
    enhanced = enhance_image(corrected if corrected is not None else img)
    _, buffer = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(req: ScanRequest):
    raw_base64 = req.image
    if "base64," in raw_base64:
        raw_base64 = raw_base64.split("base64,", 1)[1]

    try:
        processed_base64 = process_image(raw_base64)
    except Exception:
        processed_base64 = raw_base64

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{processed_base64}",
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
        "correctedImage": processed_base64,
    }
