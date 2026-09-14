"""Normalized schema for the Mendeley AI-adoption-in-Vietnamese-higher-education
dataset (pyyjfthc84). Raw column names below are copied verbatim from the real
downloaded file (see docs/dataset-schema-notes.md for the discovery process and
the construct-mapping rationale) -- everything downstream imports from here
rather than touching raw column names directly.

Discovery notes (see docs/dataset-schema-notes.md in the parent repo for full
detail):
- N = 59 respondents. Small sample -- treat all downstream statistics as
  illustrative, not inferential.
- There is no `faculty`/department field in the source data. The only
  demographic columns are role/position, highest education level, age group,
  and gender. `role` (student/staff) is the only demographic used downstream.
- SI and FC have no direct survey items -- both are proxied from the closest
  available item blocks (institutional-endorsement items for SI, reverse-coded
  concern/barrier items for FC). This is a researcher judgment call in the
  absence of a published codebook.
- Likert responses are stored as label strings, e.g. "4 (Đồng ý)", not plain
  integers -- use `parse_likert_value` before averaging.
"""

RAW_TO_ITEM_COLUMNS: dict[str, list[str]] = {
    'pe': [
        'Sử dụng công nghệ AI sẽ giúp tôi hoàn thành các tác vụ hiệu quả hơn.\n\nUsing AI technology will help me accomplish tasks more efficiently.',
        'Công nghệ AI sẽ giúp tăng cường năng lực của tôi trong việc thực hiện các công tác học thuật một cách hiệu quả. \n\nAI technology will enhance my ability to perform academic tasks effectively.',
        'Tôi tin rằng công nghệ AI sẽ giúp cải thiện chất lượng học tập của tôi.\n\nI believe AI technology will improve the quality of my learning outcomes.',
        'Việc sử dụng công nghệ AI sẽ giúp cải thiện năng lực học thuật. \n\nThe use of AI technology will lead to better academic performance.',
        'Công nghệ AI sẽ đóng góp vào một trải nghiệm học tập hiệu quả và thành công hơn.\n\nAI technology will contribute to a more productive and successful learning experience.',
    ],
    'ee': [
        'Tôi có thể dễ dàng hiểu cách tương tác với AI.\n\nMy interaction with the AI is easy for me to understand.',
        'AI cho tôi những chỉ dẫn hữu ích trong quá trình làm việc.\n\nAI provides helpful guidance in performing tasks.',
        'AI giúp tôi làm việc tốt hơn.\n\nAI will give me greater control of my work.',
        'Việc trở nên thành thạo hơn trong công việc của mình thì dễ dàng với tôi hơn là trong việc sử dụng AI. \n\nIt will be easier for me to become skillful than using AI Technology.',
        'Nhìn chung, tôi thấy AI dễ sử dụng.\n\nOverall, I will find AI easy to use.',
    ],
    'si': [
        'Các tổ chức giáo dục đã sẵn sàng để sử dụng công nghệ AI trong các chương trình học của họ. \n\nInstitutions are prepared to use AI technology in their educational programs.',
        'Các tổ chức giáo dục đã sẵn sàng để hiện đại hoá các nền tảng học tập bằng cách sử dụng AI.\n\nInstitutions are prepared to modernize their educational platforms and use AI in them.',
        'Việc ứng dụng AI trong giáo dục đại học có thể khiến giáo dục có tính tương tác cao hơn.\n\nApplication of AI in highereducation will make education more interactive.',
        'Việc ứng dụng AI trong giáo dục đại học có thể khiến hoạt động dạy - học trở nên thú vị hơn.\n\nApplication of AI in higher education will make the teaching– learning activity more interesting.',
        'Việc ứng dụng AI trong giáo dục đại học có thể làm giáo dục hiệu quả hơn về mặt chi phí.\n\nApplication of AI in higher education will make it cost-effective.',
        'Việc ứng dụng AI trong giáo dục đại học có thể làm giáo dục hiệu quả hơn về mặt thời gian.\n\nApplication of AI in higher education will save time for students and teachers.',
    ],
    'fc': [
        'Tôi lo sợ rằng việc áp dụng AI sẽ làm tăng chi phí vận hành của tổ chức.\n\nI am concerned that adopting AI technology in education might increase educational institutions’ costs.',
        'Tôi lo sợ rằng những công cụ ứng dụng AI trong giáo dục có thể làm tăng chi phí của học sinh và gia đình. \n\nI worry that AI-driven educational tools could require additional expenses for students and families.',
        'Tôi đắn đo về việc giáo dục tích hợp AI có thể dẫn đến việc suy giảm chất lượng học liệu so với các phương pháp truyền thống. \n\nI am concerned that AI driven education might result in a lower quality of learning materials compared to traditional methods.',
        'Tôi đắn đo rằng những công cụ giáo dục tích hợp AI có thể không đáp ứng được các nhu cầu học hỏi của cá nhân một cách hiệu quả. \n\nI worry that AI-powered educational tools might not effectively cater to individual learning preferences and needs.',
        'Tôi lo lắng rằng giáo dục tích hợp AI có thể tiêu tốn nhiều thời gian hơn để chuẩn bị và sử dụng một cách hiệu quả.\n\nI am concerned that AI-driven education might require more time to set up and use effectively.',
        'Tôi lo lắng rằng việc giáo dục dựa trên AI có thể ít thuận tiện hơn so với những phương pháp truyền thống bởi do các vấn đề hoặc sự phức tạp về mặt kĩ thuật.\n\nI worry that AI-based learning might be less convenient than traditional methods due to technical challenges or complexities.',
    ],
    'bi': [
        'Tôi hứng thú trong việc tiếp tục xem những chương trình giáo dục có sự tham gia của các trợ giảng hoặc giảng viên AI. \n\nI am interested in continuing to watch educational programs facilitated by AI-driven teaching assistants or instructors.',
        'Tôi tự tin rằng tôi sẽ thường xuyên sử dụng các công cụ giáo dục ứng dụng AI cho việc học tập và làm việc trong tương lai.\n\nI am confident that I will frequently utilize AI-driven educational tools for learning in the future.',
        'So với các phương pháp truyền thống, tôi ưu tiên việc học tập và làm việc trên những nền tảng có ứng dụng AI. \n\nIn comparison to traditional teaching methods, I prefer learning through AI-driven educational platforms.',
        'Nếu được trao cơ hội, tôi sẽ gợi ý các công cụ giáo dục và nguồn học liệu ứng dụng AI cho các sinh viên và giảng viên khác.\n\nIf given the chance, I would recommend AI-based educational tools and resources to fellow students and educators.',
        'Tôi rất chào đón việc sử dụng công nghệ AI để cải thiện trải nghiệm học tập và làm việc của bản thân ở bậc đại học.\n\nI am open to the idea of using AI technology to enhance my learning experience in higher education.',
    ],
}

RAW_DEMOGRAPHIC_COLUMNS: dict[str, str] = {
    'role': 'Chức danh',
}

REVERSE_CODED_CONSTRUCTS: set[str] = {'fc'}

ROLE_VALUE_MAP: dict[str, str] = {
    'Giảng viên': 'staff',
    'Sinh viên': 'student',
}

LIKERT_MAX: int = 5

NORMALIZED_COLUMNS: list[str] = [
    "respondent_id", "role",
    "pe_score", "ee_score", "si_score", "fc_score", "behavioral_intention",
]


def parse_likert_value(raw) -> int:
    """Raw Likert cells are strings like '4 (Đồng ý)' -- extract the leading int."""
    return int(str(raw).split(" ")[0])
