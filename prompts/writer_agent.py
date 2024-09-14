write_chapter_prompt="""당신은 세계적으로 유명한 베스트셀러 작가입니다. 
지금 장편 소설의 한 챕터를 작성하고 있습니다. 
이 챕터는 매우 길고 상세해야 하며, 적어도 5000단어 이상이어야 합니다.

주제: {topic}
전체 챕터 개요: {chatper_outline}
현재 작성 중인 챕터: {chapter_title}

지침:
1. 챕터 시작에 짧은 인용구로 분위기를 설정하세요.
2. 세계관을 풍부하게 발전시키세요. 역사, 문화, 사회 구조, 기술 등의 세부사항을 자연스럽게 포함하세요.
3. 캐릭터의 내적 갈등, 동기, 감정 변화를 깊이 있게 탐구하세요.
4. 주요 플롯을 진전시키면서 흥미로운 서브플롯도 발전시키세요. 챕터 끝에 반전이나 미스터리를 추가하세요.
5. 장면, 인물, 사건을 생생하게 묘사하세요. 다섯 가지 감각을 모두 활용하여 묘사하세요.
6. 서술, 대화, 내적 독백, 회상 등 다양한 문체를 사용하세요.
7. 소설의 주요 테마를 발전시키고, 관련 상징이나 모티프를 포함하세요.
8. 이전 챕터와의 연속성을 유지하고, 다음 챕터로 자연스럽게 이어지게 하세요.

이 지침을 바탕으로 '{chapter_title}' 챕터를 최대한 길고 상세하게 작성하세요. 
독자가 이 세계에 완전히 몰입할 수 있도록, 풍부한 세부사항과 깊이 있는 내용으로 채워주세요.
짧은 챕터는 절대 허용되지 않습니다. 
당신의 목표는 독자가 이 한 챕터만으로도 몇 시간 동안 푹 빠져들 수 있을 만큼 길고 매력적인 내용을 만들어내는 것입니다.
"""

review_and_edit_chapters_prompt="""
당신은 숙련된 편집자입니다. 다음 두 챕터를 주의 깊게 검토하고 필요한 편집을 수행해주세요. 
책 제목: {book_title}
전체 주제: {book_topic}

챕터 1: {chapter1_title}
{chapter1_content}

챕터 2: {chapter2_title}
{chapter2_content}

다음 사항들을 고려하여 두 챕터를 검토하고 편집해주세요:

1. 연속성: 두 챕터 간의 연결이 자연스러운지 확인하세요. 필요하다면 전환 문구를 추가하거나 수정하세요.
2. 일관성: 캐릭터, 설정, 스타일이 일관되게 유지되는지 확인하세요.
3. 주제 적합성: 각 챕터가 책의 전체 주제와 잘 연결되는지 확인하세요.
4. 흐름: 이야기의 흐름이 자연스러운지, 독자의 관심을 유지할 수 있는지 확인하세요.
5. 문체: 문장 구조, 단어 선택, 톤이 적절한지 확인하고 필요하다면 개선하세요.
6. 오류 수정: 문법, 맞춤법, 구두점 오류를 수정하세요.

편집된 두 챕터를 제공해주세요. 중요한 변경사항이 있다면 간단히 설명해주세요.
"""

plan_chapters_prompt="""
I need you to help me break down the following long-form writing instruction into multiple subtasks. 
Each subtask will guide the writing of one paragraph in the essay, and should include the main points.

The writing title and description is as follows:
{topic}
{description}

Please break it down in the following format, with each subtask taking up one line:
Paragraph 1 - [Describe the main point of the paragraph, in detail]
Paragraph 2 - [Describe the main point of the paragraph, in detail]
...

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction. 
Do not split the subtasks too finely.
As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks.
작성된 paragraph는 모두 한글로 작성되어야 합니다.
"""