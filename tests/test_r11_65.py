"""Round 11, task 65: data filter value absent from the file, month outside 1-12."""
from harness.solvers import data_solver
from harness.tasks import Task

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CSV = "order_id,region,product,month,units,price\n1,west,widget,5,5,33\n2,north,gizmo,6,28,28\n3,north,gizmo,1,39,17\n"


def _data(tmp_path, question):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "workspace" / "sales.csv").write_text(CSV, encoding="utf-8")
    ins = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). " + question
           + " Use bash with python3 to compute it." + TAIL)
    return Task("t", "data_analysis", ins, 3.0, "exact_int", d, d / "workspace")


def _status(res):
    return res.status, res.answer


def test_product_outside_the_file_and_the_known_words_has_no_answer(tmp_path):
    res = data_solver.solve(_data(tmp_path, "How many units of product 'tesseract' were sold in months 5 through 6 inclusive?"))
    assert res.status == "infeasible" and "does not occur" in res.note


def test_known_word_missing_from_a_small_file_is_an_honest_zero(tmp_path):
    q = "How many units of product 'sprocket' were sold in months 5 through 6 inclusive?"
    assert _status(data_solver.solve(_data(tmp_path, q))) == ("solved", "0")


def test_empty_month_window_of_a_present_product_stays_zero(tmp_path):
    q = "How many units of product 'widget' were sold in months 7 through 8 inclusive?"
    assert _status(data_solver.solve(_data(tmp_path, q))) == ("solved", "0")


def test_region_outside_the_file_with_a_product_filter(tmp_path):
    q = "How many units of product 'widget' were sold in region 'central' in months 5 through 6 inclusive?"
    assert data_solver.solve(_data(tmp_path, q)).status == "infeasible"
    q = "How many units of product 'gizmo' were sold in region 'west' in months 5 through 6 inclusive?"
    assert _status(data_solver.solve(_data(tmp_path, q))) == ("solved", "0")


def test_value_present_only_in_other_rows_is_not_absent(tmp_path):
    # 'west' occurs in the file (row of another product): an honest 0, not infeasible
    q = "How many units of product 'gizmo' were sold in region 'west'?"
    assert _status(data_solver.solve(_data(tmp_path, q))) == ("solved", "0")


def test_month_13_of_a_recognized_question_is_infeasible(tmp_path):
    q = "How many units of product 'widget' were sold in months 13 through 14 inclusive?"
    res = data_solver.solve(_data(tmp_path, q))
    assert res.status == "infeasible" and "month outside" in res.note
    q = "How many units of product 'widget' were returned in months 13 through 14 inclusive?"
    assert data_solver.solve(_data(tmp_path, q)).status == "unparsed"


def _file_task(tmp_path, csv_text, question):
    t = _data(tmp_path, question)
    (t.workspace / "sales.csv").write_text(csv_text, encoding="utf-8")
    return t


def test_region_collection_covers_all_rows_with_a_value_outside_the_known_words(tmp_path):
    csv_text = CSV + "4,central,widget,2,1,1\n"
    q = "How many units of product 'gizmo' were sold in region 'central' in months 5 through 6 inclusive?"
    assert _status(data_solver.solve(_file_task(tmp_path, csv_text, q))) == ("solved", "0")
    q = "How many units of product 'gizmo' were sold in region 'orbital' in months 5 through 6 inclusive?"
    assert data_solver.solve(_file_task(tmp_path, csv_text, q)).status == "infeasible"


def test_month_13_is_a_real_month_of_a_two_year_file(tmp_path):
    csv_text = CSV + "4,west,widget,14,9,10\n"
    q = "How many units of product 'widget' were sold in months 13 through 14 inclusive?"
    assert _status(data_solver.solve(_file_task(tmp_path, csv_text, q))) == ("solved", "9")
    # a file of 1..12 only: no such month
    assert data_solver.solve(_data(tmp_path / "x", q)).status == "infeasible"
    # no workspace: the rule stays
    assert data_solver.parse(_data(tmp_path / "y", q).instruction).status == "infeasible"


def test_plural_or_singular_of_a_value_in_the_file_goes_to_the_model(tmp_path):
    q = "How many units of product 'gizmos' were sold in months 5 through 6 inclusive?"
    assert data_solver.solve(_data(tmp_path, q)).status == "unparsed"
    q = "How many units of product 'tesseracts' were sold in months 5 through 6 inclusive?"
    assert data_solver.solve(_data(tmp_path / "z", q)).status == "infeasible"
