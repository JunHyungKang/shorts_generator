import os
import re
from typing import Any, Dict, Iterator, List, Tuple, TypedDict

import streamlit_mermaid as stmd
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langgraph.graph import END, Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from pydantic import BaseModel, Field

from prompts.writer_agent import (
    plan_chapters_prompt,
    review_and_edit_chapters_prompt,
    write_chapter_prompt,
)


class Topic(BaseModel):
    title: str = Field(description="Generated topic title")
    description: str = Field(description="Brief description of the topic")

class TopicList(BaseModel):
    topics: List[Topic]

class Chapter(BaseModel):
    title: str = Field(description="Chapter title")
    content: str = Field(description="Chapter content")
    
class ChapterList(BaseModel):
    chapters: List[Chapter]

class ChapterOutline(BaseModel):
    paragraphs: List[str] = Field(description="전체 주제에 따라 구성한 챕터 목록")

class ChapterEvaluation(BaseModel):
    score: float = Field(description="Quality and relevance score from 0 to 1")
    reason: str = Field(description="Explanation for the given score")

class Book(BaseModel):
    title: str = Field(description="Book title")
    chapters: List[Chapter] = Field(description="List of chapters")

class WriteState(TypedDict):
    task: str
    steps: List[str]
    results: Dict[str, Any]
    result: str
    
class WriterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self.define_graph()
        self.show_graph()

    def generate_topics(self, audience_info: str) -> TopicList:
        prompt = ChatPromptTemplate.from_template(
            "Given the following audience information, generate 5 interesting topic ideas:\n{audience_info}"
        )
        llm = self.llm.with_structured_output(TopicList)
        topics_chain = prompt | llm
        return topics_chain.invoke({"audience_info": audience_info})

    def select_best_topic(self, topics: List[Topic]) -> Topic:
        prompt = ChatPromptTemplate.from_template(
            "From the following topics, select the most interesting one:\n{topics}"
        )
        llm = self.llm.with_structured_output(Topic)
        selection_chain = prompt | llm
        return selection_chain.invoke({"topics": "\n".join([f"{t.title}: {t.description}" for t in topics.topics])})

    def plan_chapters(self, topic: Topic) -> List[str]:        
        prompt = ChatPromptTemplate.from_template(
            plan_chapters_prompt
        )
        llm = self.llm.with_structured_output(ChapterOutline)
        
        planning_chain = prompt | llm
        
        result = planning_chain.invoke({
            "topic": topic.title,
            "description": topic.description
        })
        
        return result

    def write_chapter(self, topic: Topic, outlines, chapter_title: str) -> Chapter:
        
        prompt = ChatPromptTemplate.from_template(write_chapter_prompt)
        llm = self.llm.with_structured_output(Chapter)
        writing_chain = prompt | llm
        return writing_chain.invoke({"topic": topic.title, 
                                     "chatper_outline": outlines, 
                                     "chapter_title": chapter_title})
    
    def evaluate_chapter(self, chapter: Chapter, topic: Topic) -> float:
        prompt = ChatPromptTemplate.from_template(
            "Evaluate the following chapter for the book on '{topic}':\n\nTitle: {title}\n\nContent: {content}\n\n"
            "Rate the chapter's quality and relevance from 0 to 1, where 1 is excellent and 0 is poor."
        )
        llm = self.llm.with_structured_output(ChapterEvaluation)
        evaluation_chain = prompt | llm
        result = evaluation_chain.invoke({"topic": topic.title, "title": chapter.title, "content": chapter.content})
        return float(result.score)

    def review_and_edit_chapters(self, book: Book, start_index: int = 0) -> List[Chapter]:
        prompt = ChatPromptTemplate.from_template(review_and_edit_chapters_prompt)
        llm = self.llm.with_structured_output(ChapterList)
        editing_chain = prompt | llm
        
        edited_chapters = book.chapters[:start_index]  # 시작 인덱스 이전의 챕터들은 그대로 유지
        
        for i in range(start_index, len(book.chapters) - 1):
            chapter1 = book.chapters[i]
            chapter2 = book.chapters[i + 1]
            
            result = editing_chain.invoke({
                "book_title": book.title,
                "book_topic": book.title,  # 필요하다면 별도의 'topic' 필드를 Book 모델에 추가할 수 있습니다.
                "chapter1_title": chapter1.title,
                "chapter1_content": chapter1.content,
                "chapter2_title": chapter2.title,
                "chapter2_content": chapter2.content
            }).chapters
            
            if i == start_index:
                edited_chapters.extend(result)  # 첫 번째 반복에서는 두 챕터 모두 추가
            else:
                edited_chapters[-1] = result[0]  # 이전 챕터를 업데이트
                edited_chapters.append(result[1])  # 새로운 챕터 추가
        
        # 마지막 챕터 처리 (이전 챕터와 함께 다시 검토)
        if len(book.chapters) > 1:
            last_result = editing_chain.invoke({
                "book_title": book.title,
                "book_topic": book.title,
                "chapter1_title": edited_chapters[-1].title,
                "chapter1_content": edited_chapters[-1].content,
                "chapter2_title": book.chapters[-1].title,
                "chapter2_content": book.chapters[-1].content
            }).chapters
            print('last_result\n',last_result)
            print('edited_chapters\n',edited_chapters)
            edited_chapters[-1] = last_result[0]  # 마지막에서 두 번째 챕터 업데이트
            edited_chapters.append(last_result[1])  # 마지막 챕터 추가
        elif len(book.chapters) == 1 and start_index == 0:
            # 책에 챕터가 하나밖에 없는 경우
            edited_chapters.append(book.chapters[0])
        
        return edited_chapters
    
    def review_and_edit(self, book: Book) -> Book:
        edited_chapters = self.review_and_edit_chapters(book, 0)
        return Book(title=book.title, chapters=edited_chapters)

    def define_graph(self):
        graph = StateGraph(WriteState)

        def generate_topic_ideas(state):
            topics = self.generate_topics(state["task"])
            state["results"]["topics"] = topics
            state["steps"].append("Generated topic ideas")
            return state

        def choose_topic(state):
            best_topic = self.select_best_topic(state["results"]["topics"])
            state["results"]["selected_topic"] = best_topic
            state["steps"].append("Chose best topic")
            return state

        def create_outline(state):
            chapters = self.plan_chapters(state["results"]["selected_topic"])
            state["results"]["chapter_outline"] = chapters
            state["steps"].append("Created chapter outline")
            return state

        # def write_chapters(state):
        #     topic = state["results"]["selected_topic"]
        #     outlines = '\n'.join(state["results"]["chapter_outline"].paragraphs)
        #     chapters = []
        #     for chapter_title in state["results"]["chapter_outline"].paragraphs:
        #         chapter = self.write_chapter(topic, outlines, chapter_title)
        #         chapters.append(chapter)
        #     state["results"]["draft_chapters"] = chapters
        #     state["steps"].append("Wrote chapters")
        #     return state
        
        def write_chapters(state):
            topic = state["results"]["selected_topic"]
            outlines = '\n'.join(state["results"]["chapter_outline"].paragraphs)
            chapters = []
            for chapter_title in state["results"]["chapter_outline"].paragraphs:
                chapter = self.write_chapter(topic, outlines, chapter_title)
                chapters.append(chapter)
            state["results"]["final_book"] = f'# {topic.title}\n\n' + '\n'.join([f'## {x.title}\n{x.content}' for x in chapters])
            state["steps"].append("Wrote chapters")
            return state

        def evaluate_chapters(state):
            topic = state["results"]["selected_topic"]
            chapters = state["results"]["draft_chapters"]
            evaluations = [self.evaluate_chapter(chapter, topic) for chapter in chapters]
            state["results"]["chapter_evaluations"] = evaluations
            state["steps"].append("Evaluated chapters")
            return state

        def rewrite_chapters(state):
            topic = state["results"]["selected_topic"]
            chapters = state["results"]["draft_chapters"]
            evaluations = state["results"]["chapter_evaluations"]
            for i, (chapter, evaluation) in enumerate(zip(chapters, evaluations)):
                if evaluation < 0.7:  # 임계값 설정
                    new_chapter = self.write_chapter(topic, chapter.title)
                    chapters[i] = new_chapter
            state["results"]["draft_chapters"] = chapters
            state["steps"].append("Rewrote low-quality chapters")
            return state

        def edit_book(state):
            topic = state["results"]["selected_topic"]
            chapters = state["results"]["draft_chapters"]
            book = Book(title=topic.title, chapters=chapters)
            edited_book = self.review_and_edit(book)
            state["results"]["final_book"] = [f'# {edited_book.title}\n\n## {x.title}\n{x.content}' for x in edited_book.chapters]
            state["steps"].append("Edited book")
            state["result"] = f"Completed book: {edited_book.title}"
            return state

        graph.add_node("generate_topic_ideas", generate_topic_ideas)
        graph.add_node("choose_topic", choose_topic)
        graph.add_node("create_outline", create_outline)
        graph.add_node("write_chapters", write_chapters)
        # graph.add_node("evaluate_chapters", evaluate_chapters)
        # graph.add_node("rewrite_chapters", rewrite_chapters)
        # graph.add_node("edit_book", edit_book)

        graph.set_entry_point("generate_topic_ideas")
        graph.add_edge("generate_topic_ideas", "choose_topic")
        graph.add_edge("choose_topic", "create_outline")
        graph.add_edge("create_outline", "write_chapters")
        # graph.add_edge("write_chapters", "evaluate_chapters")
        graph.add_edge("write_chapters", END)

        # def should_rewrite(state):
        #     evaluations = state["results"]["chapter_evaluations"]
        #     return "rewrite_chapters" if any(eval < 0.7 for eval in evaluations) else "edit_book"

        # graph.add_conditional_edges(
        #     "evaluate_chapters",
        #     should_rewrite,
        #     {
        #         "rewrite_chapters": "rewrite_chapters",
        #         "edit_book": "edit_book"
        #     }
        # )
        # graph.add_edge("rewrite_chapters", "evaluate_chapters")
        # graph.add_edge("edit_book", END)

        return graph.compile()

    def run(self, audience_info: str) -> Tuple[Book, List[Dict[str, Any]]]:
        initial_state: WriteState = {
            "task": audience_info,
            "steps": [],
            "results": {},
            "result": ""
        }
        
        final_book = None
        
        for state in self.graph.stream(initial_state):
            print('STATE\n',state)
            for key, value in state.items():
                if "final_book" in value["results"]:
                    final_book = value["results"]["final_book"]

        return final_book
    
    def show_graph(self):
        graph_mermaid = self.graph.get_graph().draw_mermaid()
        graph_mermaid = graph_mermaid.replace(
            "%%{init: {'flowchart': {'curve': 'linear'}}}%%", ""
        ).replace("graph TD;", "graph LR;")
        # stmd.st_mermaid(graph_mermaid)
        return graph_mermaid
