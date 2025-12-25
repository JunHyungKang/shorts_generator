# Shorts Generator (AI Auto-Pilot)

**Shorts Generator**는 시니어(6070) 타겟의 유튜브 트렌드를 분석하는 것에서 나아가, **주제 선정부터 영상 제작, 그리고 유튜브 자동 업로드까지 스룹(Throughput) 전체를 자동화**하는 AI 시스템입니다.

**Multi-Agent Ecosystem**을 기반으로 각자 역할을 가진 에이전트들이 협업하여 최적의 콘텐츠를 생산합니다.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![React](https://img.shields.io/badge/react-18-blue.svg)

## 🤖 Multi-Agent Architecture

이 시스템은 3개의 핵심 에이전트와 Orchestrator로 구성됩니다.

### 1. 🔭 Trend Scout Agent (트렌드 발굴)
- **Role**: 시장 조사를 담당하는 기획자
- **Tasks**:
    - DuckDuckGo 및 YouTube 검색을 통해 시니어 인기 키워드 발굴
    - LLM을 활용하여 잠재적 히트 주제 선정 및 검증
    - 콘텐츠 기획안(제목, 타겟, 키워드) 생성

### 2. 🎬 Video Creator Agent (영상 제작)
- **Role**: 영상을 실제로 제작하는 PD/편집자
- **Tasks**:
    - 기획안을 바탕으로 매력적인 스크립트 작성
    - 스크립트에 어울리는 이미지 생성 (AI) 또는 영상 소스 검색
    - TTS(Text-to-Speech) 생성 및 배경음악 믹싱
    - FFmpeg를 활용한 숏폼(Shorts) 영상 렌더링

### 3. 🎼 Orchestrator (관리 및 조율)
- **Role**: 전체 파이프라인을 관리하는 총괄 PM
- **Tasks**:
    - Trend Scout와 Video Creator 간의 워크플로우 제어 (LangGraph)
    - 에러 핸들링 및 재시도 로직 수행
    - 최종 결과물의 품질 검수

### ➕ Post-Processing (자동 업로드)
- 제작 완료된 영상을 YouTube Data API를 통해 채널에 자동으로 업로드합니다.
- 제목, 설명, 태그, 썸네일 등을 최적화하여 게시합니다.

## 🛠️ 기술 스택 (Tech Stack)

### Backend (Python)
- **Framework**: FastAPI
- **Agent Orchestration**: LangGraph, LangChain
- **LLM**: Gemini 2.0 Flash (via OpenRouter), Local LLMs (Ollama)
- **Tools**: `duckduckgo-search`, `youtube-search-python`, `ffmpeg-python`, `google-api-python-client`

### Frontend (React)
- **Framework**: React (Vite)
- **UI**: TailwindCSS

## 📂 프로젝트 구조 (Project Structure)

Backend와 Frontend가 분리된 Monorepo 구조입니다.

```
shorts/
├── backend/             # 🐍 Python Backend Core
│   ├── src/
│   │   ├── agents/      # [Agents Hub]
│   │   │   ├── trend_scout/    # 트렌드 발굴 에이전트
│   │   │   ├── video_creator/  # 영상 생성 에이전트
│   │   │   └── orchestrator/   # 워크플로우 관리자
│   │   └── api/         # FastAPI 엔드포인트
│   ├── tests/           # 유닛 및 통합 테스트
│   └── pyproject.toml
│
├── frontend/            # ⚛️ React Dashboard
│   ├── src/
│   └── package.json
│
└── README.md
```

## 🚀 시작하기 (Getting Started)

### 사전 요구 사항
- Python 3.11+
- Node.js 18+
- API Keys: OpenRouter, YouTube Data API

### Backend 실행
```bash
cd backend
uv sync
# 서버 실행
uv run python -m uvicorn src.api.main:app --reload
```

### Frontend 실행
```bash
cd frontend
npm install
npm run dev
```

---
*Created by [JunHyungKang/shorts_generator]*
