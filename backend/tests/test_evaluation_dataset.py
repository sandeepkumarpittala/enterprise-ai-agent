
from app.evaluation.dataset import EVALUATION_DATASET


def test_dataset_has_fifteen_unique_cases():
    assert len(EVALUATION_DATASET) == 15
    ids = [case["id"] for case in EVALUATION_DATASET]
    assert len(ids) == len(set(ids))


def test_dataset_contains_answerable_and_unanswerable_cases():
    answerable = [
        case for case in EVALUATION_DATASET if case["answerable"]
    ]
    unanswerable = [
        case for case in EVALUATION_DATASET if not case["answerable"]
    ]

    assert len(answerable) == 11
    assert len(unanswerable) == 4


def test_answerable_cases_have_expected_answers_and_sources():
    answerable = [
        case for case in EVALUATION_DATASET if case["answerable"]
    ]

    for case in answerable:
        assert case["expected_answer"]
        assert case["expected_source"]


def test_unanswerable_cases_have_no_expected_answer_or_source():
    unanswerable = [
        case for case in EVALUATION_DATASET if not case["answerable"]
    ]

    for case in unanswerable:
        assert case["expected_answer"] is None
        assert case["expected_source"] is None


def test_every_case_has_required_fields():
    required_fields = {
        "id",
        "question",
        "expected_answer",
        "expected_source",
        "answerable",
    }

    for case in EVALUATION_DATASET:
        assert required_fields.issubset(case.keys())
        assert case["id"]
        assert case["question"]
        assert isinstance(case["answerable"], bool)
