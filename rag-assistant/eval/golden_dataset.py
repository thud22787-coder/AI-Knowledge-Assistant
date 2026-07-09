"""Golden dataset for future RAGAS evaluation.

The samples are grounded in test documents that already exist in the repo's
test suite, so we avoid inventing knowledge outside uploaded system data.
"""

GOLDEN_DATASET = [
    {
        "question": "Trong sample PDF, câu ở trang 1 là gì?",
        "ground_truth": "Page one: the quick brown fox jumps over the lazy dog.",
        "source_doc": "tests/test_parser.py::sample.pdf",
    },
    {
        "question": "Trong sample PDF, trang 2 nói câu nào?",
        "ground_truth": "Page two: pack my box with five dozen liquor jugs.",
        "source_doc": "tests/test_parser.py::sample.pdf",
    },
    {
        "question": "Đoạn đầu tiên trong sample DOCX là gì?",
        "ground_truth": "DOCX paragraph one: alpha beta gamma.",
        "source_doc": "tests/test_parser.py::sample.docx",
    },
    {
        "question": "Đoạn thứ hai trong sample DOCX là gì?",
        "ground_truth": "DOCX paragraph two: delta epsilon zeta.",
        "source_doc": "tests/test_parser.py::sample.docx",
    },
    {
        "question": "Đoạn thứ ba trong sample DOCX là gì?",
        "ground_truth": "DOCX paragraph three: eta theta iota.",
        "source_doc": "tests/test_parser.py::sample.docx",
    },
    {
        "question": "File TXT dùng để test xử lý document nói gì ở đoạn đầu?",
        "ground_truth": (
            "Paragraph one has enough text to make the parser and chunker work properly. "
            "It should become at least one chunk."
        ),
        "source_doc": "tests/test_documents.py::process.txt",
    },
    {
        "question": "Trong file TXT process.txt, đoạn thứ hai nói gì?",
        "ground_truth": (
            "Paragraph two also has enough content to ensure multiple paragraphs are handled. "
            "This gives us more than one paragraph for the test."
        ),
        "source_doc": "tests/test_documents.py::process.txt",
    },
    {
        "question": "Nội dung của file PDF process.pdf là gì?",
        "ground_truth": "This is a PDF document for processing.",
        "source_doc": "tests/test_documents.py::process.pdf",
    },
    {
        "question": "Đoạn đầu tiên trong file DOCX process.docx là gì?",
        "ground_truth": "This is the first DOCX paragraph for processing. It should be parsed into text and chunked.",
        "source_doc": "tests/test_documents.py::process.docx",
    },
    {
        "question": "Đoạn thứ hai trong file DOCX process.docx là gì?",
        "ground_truth": (
            "This is the second DOCX paragraph for processing. "
            "It gives the test another paragraph to verify."
        ),
        "source_doc": "tests/test_documents.py::process.docx",
    },
]
