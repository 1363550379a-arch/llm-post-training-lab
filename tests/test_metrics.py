from src.metrics import gsm_gold,pred_number,num_equal,normalize_math_answer

def test_gold():
    assert gsm_gold("thought <<1+2=3>> done #### 1,234") == "1234"
    assert num_equal(pred_number("we calculate 3. Final answer: 42"),"42")
    assert pred_number("answer is 4 with no final marker") is None
    assert normalize_math_answer("a <<1+2=3>> b")=="a  b"

def test_wrong():
    assert not num_equal("3","4")
    assert not num_equal(None,"4")
