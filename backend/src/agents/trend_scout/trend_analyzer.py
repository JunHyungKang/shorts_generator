import json
import re
from typing import List, Dict
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from src.utils.openrouter import get_chat_model

from pydantic import BaseModel, Field

class KeywordList(BaseModel):
    keywords: List[str] = Field(description="List of 5 specific search keywords extracted from the text")

class TrendAnalyzer:
    def __init__(self):
        # Use Gemini 2.0 Flash as primary for better structured output support
        self.llm = get_chat_model(model_name="google/gemini-2.0-flash-exp:free", temperature=0.5)
        self.ddgs = DDGS()

    def discover_keywords(self) -> List[str]:
        """
        Phase 1: Discover trending keywords for seniors using Web Search + LLM.
        """
        print("🔍 Phase 1: Discovering Trends via Web Search...")
        
        # 1. Search for broad topics
        queries = [
            "2024년 60대 70대 인기 검색어 트렌드",
            "요즘 시니어들에게 인기 있는 취미 생활",
            "한국 노년층 유튜브 인기 주제",
            "중장년층 핫한 키워드"
        ]
        
        search_results = []
        for q in queries:
            try:
                results = self.ddgs.text(q, max_results=3)
                if results:
                    search_results.extend([r['body'] for r in results])
            except Exception as e:
                print(f"   ⚠️ Search failed for '{q}': {e}")
        
        context = "\n".join(search_results)
        if not context:
            print("⚠️ No search results found. Using default context.")
            context = "시니어들은 최근 건강, 재테크, 트로트, 여행에 관심이 많습니다."
        
        # 2. Extract specific keywords using Structured Output
        print("   - Asking LLM for structured keywords...")
        try:
            structured_llm = self.llm.with_structured_output(KeywordList)
            
            prompt = ChatPromptTemplate.from_template("""
            다음은 '한국 시니어(60대~70대) 트렌드'에 대한 최근 웹 검색 결과입니다.
            요즘 시니어들이 유튜브에서 찾아볼 만한 '구체적인 검색 키워드' 5개를 추출해주세요.
            
            [검색 결과]
            {context}
            """)
            
            chain = prompt | structured_llm
            result = chain.invoke({"context": context})
            
            if result and result.keywords:
                 print(f"✅ Extracted Keywords: {result.keywords}")
                 return result.keywords
            else:
                 raise ValueError("Empty result from structured output")
                 
        except Exception as e:
            print(f"⚠️ Structured output failed ({e}). Falling back to manual parsing.")
            # Fallback logic could be here, or just return defaults
            print("   - Using default fallback keywords.")
            return ["시니어 건강", "트로트 인기곡", "60대 취미", "노년 재테크", "황토길 맨발걷기"]

    def search_youtube_videos(self, keywords: List[str], max_videos: int = 3) -> List[Dict]:
        """
        Phase 2: Search YouTube for the discovered keywords.
        """
        print(f"📹 Phase 2: Searching YouTube for {len(keywords)} keywords...")
        
        all_videos = []
        
        for keyword in keywords:
            print(f"   - Searching: {keyword}")
            try:
                videos_search = VideosSearch(keyword, limit=max_videos)
                results = videos_search.result()
                
                if 'result' in results:
                    for v in results['result']:
                        video_data = {
                            "search_keyword": keyword,
                            "title": v.get('title'),
                            "views": v.get('viewCount', {}).get('short', 'N/A'),
                            "duration": v.get('duration'),
                            "channel": v.get('channel', {}).get('name'),
                            "link": v.get('link')
                        }
                        all_videos.append(video_data)
            except Exception as e:
                print(f"⚠️ Error searching for '{keyword}': {e}")
                continue
        
        print(f"✅ Collected {len(all_videos)} videos.")
        return all_videos

    def analyze_trends(self, videos: List[Dict]) -> str:
        """
        Phase 3: Generate a trend report using LLM.
        """
        print("📊 Phase 3: Analyzing Trends...")
        
        # Simplify video data for LLM context to save tokens
        video_context = ""
        for v in videos:
            video_context += f"- [{v['search_keyword']}] {v['title']} (Views: {v['views']}, Channel: {v['channel']})\n"
            
        prompt = ChatPromptTemplate.from_template("""
        당신은 '유튜브 트렌드 분석가'입니다.
        아래는 최근 시니어(6070) 타겟의 인기 키워드로 검색된 유튜브 영상 리스트입니다.
        
        이 데이터를 분석하여 다음 내용을 포함한 '시니어 유튜브 트렌드 보고서'를 작성해주세요:
        1. **주요 관심사 요약**: 어떤 주제들이 가장 인기가 많은가?
        2. **콘텐츠 특징**: 제목이나 주제에서 발견되는 공통된 패턴은? (예: 자극적인 썸네일 제목, "절대 하지 마라" 등의 경고형 제목 등)
        3. **크리에이터 팁**: 시니어 타겟 영상을 만들 때 참고할 점.

        [수집된 영상 리스트]
        {video_context}
        
        보고서는 한국어로 작성해주세요.
        """)
        
        response = self.llm.invoke(prompt.format(video_context=video_context))
        print(f"DEBUG Phase 3 Response: {response}")
        return response.content

    def run_analysis(self) -> str:
        """
        Run the full analysis pipeline.
        """
        keywords = self.discover_keywords()
        videos = self.search_youtube_videos(keywords)
        report = self.analyze_trends(videos)
        return report
