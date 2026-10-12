from lumina.evaluate import evaluate


def test_golden_set_runs_and_reports_baseline():
    report = evaluate()
    assert report["corpus_documents"] == 7
    assert report["metrics"]["hybrid"]["cases"] == 7
    assert len(report["cases"]) == 14
    assert 0 <= report["metrics"]["vector"]["recall_at_5"] <= 1
