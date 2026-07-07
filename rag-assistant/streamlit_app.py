import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{API_BASE_URL}/auth/login"
REGISTER_ENDPOINT = f"{API_BASE_URL}/auth/register"


st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if st.session_state.access_token is None:
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login")

        if login_submitted:
            try:
                login_response = requests.post(
                    LOGIN_ENDPOINT,
                    data={"username": login_email, "password": login_password},
                    timeout=60,
                )
                if login_response.status_code == 401:
                    st.error("Invalid email or password.")
                else:
                    login_response.raise_for_status()
                    st.session_state.access_token = login_response.json()["access_token"]
                    st.rerun()
            except requests.RequestException as exc:
                st.error(f"Login request failed: {exc}")
            except (KeyError, ValueError) as exc:
                st.error(f"Unexpected login response: {exc}")

    with register_tab:
        with st.form("register_form"):
            register_email = st.text_input("Email", key="register_email")
            register_password = st.text_input("Password", type="password", key="register_password")
            register_submitted = st.form_submit_button("Register")

        if register_submitted:
            try:
                register_response = requests.post(
                    REGISTER_ENDPOINT,
                    json={"email": register_email, "password": register_password},
                    timeout=60,
                )
                if register_response.status_code == 400:
                    st.error("Email already exists.")
                else:
                    register_response.raise_for_status()
                    st.success("Registration successful. Please switch to the Login tab.")
            except requests.RequestException as exc:
                st.error(f"Registration request failed: {exc}")
            except ValueError as exc:
                st.error(f"Unexpected registration response: {exc}")

    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.access_token = None
    st.rerun()

auth_headers = {"Authorization": f"Bearer {st.session_state.access_token}"}


def upload_and_process(file_obj):
    files = {
        "file": (file_obj.name, file_obj, getattr(file_obj, "type", None) or "application/octet-stream")
    }

    upload_response = requests.post(
        f"{API_BASE_URL}/documents/upload", files=files, headers=auth_headers, timeout=60
    )
    upload_response.raise_for_status()
    upload_data = upload_response.json()

    document_id = upload_data["id"]
    process_response = requests.post(
        f"{API_BASE_URL}/documents/{document_id}/process", headers=auth_headers, timeout=60
    )
    process_response.raise_for_status()
    process_data = process_response.json()

    return upload_data, process_data


def delete_document(document_id):
    delete_response = requests.delete(
        f"{API_BASE_URL}/documents/{document_id}", headers=auth_headers, timeout=60
    )
    delete_response.raise_for_status()


st.sidebar.header("Upload document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if st.sidebar.button("Upload", width="stretch"):
    if uploaded_file is None:
        st.sidebar.error("Please choose a file first.")
    else:
        try:
            upload_data, process_data = upload_and_process(uploaded_file)
            st.sidebar.success(
                f"Uploaded {upload_data['filename']} and created {process_data['chunk_count']} chunks."
            )
        except requests.RequestException as exc:
            st.sidebar.error(f"Request failed: {exc}")
        except (KeyError, ValueError) as exc:
            st.sidebar.error(f"Unexpected response: {exc}")


st.title("Documents")

try:
    documents_response = requests.get(f"{API_BASE_URL}/documents", headers=auth_headers, timeout=60)
    documents_response.raise_for_status()
    documents = documents_response.json()
except requests.RequestException as exc:
    st.error(f"Request failed: {exc}")
    documents = []
except ValueError as exc:
    st.error(f"Unexpected response: {exc}")
    documents = []

if documents:
    st.dataframe(
        [
            {
                "filename": document.get("filename"),
                "status": document.get("status"),
                "created_at": document.get("created_at"),
            }
            for document in documents
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Delete document")
    for document in documents:
        cols = st.columns([4, 1])
        cols[0].write(document.get("filename"))
        if cols[1].button("Delete", key=f"delete_{document.get('id')}"):
            try:
                delete_document(document["id"])
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
else:
    st.info("No documents found.")


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.divider()
st.header("Chat")

if st.button("Xóa hội thoại"):
    st.session_state.chat_history = []
    st.session_state.conversation_id = None
    st.rerun()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        if item["role"] == "assistant" and item.get("sources"):
            with st.expander("Sources"):
                for source in item["sources"]:
                    text = source.get("text", "")
                    st.write(
                        f"- document_id: {source.get('document_id')}, "
                        f"page_number: {source.get('page_number')}, "
                        f"text: {text[:200]}"
                    )

question = st.chat_input("Hỏi gì đó về tài liệu...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "question": question,
                "conversation_id": st.session_state.conversation_id,
            },
            headers=auth_headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        st.session_state.conversation_id = data["conversation_id"]
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
            }
        )
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
