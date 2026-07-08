from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


def test_shows_login_and_register_tabs_when_not_authenticated():
    at = AppTest.from_file("streamlit_app.py")

    at.run()

    assert len(at.tabs) == 2
    assert at.tabs[0].label == "Login"
    assert at.tabs[1].label == "Register"
    assert not at.exception


def test_login_success_shows_documents_page():
    login_response = Mock()
    login_response.status_code = 200
    login_response.json.return_value = {"access_token": "fake-token-123"}

    documents_response = Mock()
    documents_response.status_code = 200
    documents_response.json.return_value = []

    with patch("requests.post", return_value=login_response), patch(
        "requests.get", return_value=documents_response
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run()

        at.text_input[0].set_value("user@example.com")
        at.text_input[1].set_value("secret-password")
        at.button[0].click()
        at.run()

    assert not at.exception
    assert at.title[0].value == "Documents"
