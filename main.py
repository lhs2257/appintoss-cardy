import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(req: ScanRequest):
    raw_base64 = req.image
    if "base64," in raw_base64:
        raw_base64 = raw_base64.split("base64,", 1)[1]

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{raw_base64}",
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
    }
