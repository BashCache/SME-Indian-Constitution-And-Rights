# cli_client.py
import requests, sys
from getpass import getpass
import os

BASE_URL = "http://127.0.0.1:8000"

def login():
    print("=== Login ===")
    
    MAX_RETRIES = 3
    attempts = 0

    while attempts < MAX_RETRIES:
        username = input("Username: ").strip()
        password = getpass("Password: ").strip()

        resp = requests.post(f"{BASE_URL}/auth/login", json={
            "username": username, "password": password
        })

        if resp.status_code == 200:
            print(resp.json()["message"])
            return username
        else:
            attempts += 1
            print(f"❌ Invalid credentials. Attempts left: {MAX_RETRIES - attempts}")

        if attempts == MAX_RETRIES:
            print("🚫 Too many failed attempts. Exiting.")
            sys.exit(1)

def list_sessions(username):
    r = requests.get(f"{BASE_URL}/sessions/{username}")
    sessions = r.json()
    if not sessions:
        print("\nNo previous sessions found.")
    else:
        print("\n=== Existing Sessions ===")
        for i, s in enumerate(sessions, 1):
            print(f"{i}. {s['title']} (id: {s['session_id']})")
    return sessions

def create_session(username):
    title = input("Enter title for new chat: ").strip()
    r = requests.post(f"{BASE_URL}/sessions/create", json={
        "username": username, "title": title
    })
    if r.status_code == 200:
        s = r.json()
        print(f"✅ Created new session: {s['title']} ({s['session_id']})")
        return s
    else:
        print("❌ Error creating session:", r.text)
        sys.exit(1)
        
# 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
def chat_loop(session_id):
    print("\n=== Start Chatting ===")
    print("(Type 'exit' to return to main menu)")
    print("(Type 'upload' to attach a file before asking a question)\n")

    while True:
        uploaded_file = None
        msg = input("You: ").strip()
        if msg.lower() == "exit":
            print("👋 Ending chat.")
            break
        
        if msg.lower() == "upload":
            filepath = input("Enter path to file (PDF/DOCX/PPTX): ").strip()
            if not os.path.exists(filepath):
                print("❌ File not found.")
                continue

            print(f"📤 Uploading {filepath}...")
            with open(filepath, "rb") as f:
                files = {"file": f}
                data = {"username": session_id.split('_')[0], "session_id": session_id}
                r = requests.post(f"{BASE_URL}/upload", files=files, data=data)

            if r.status_code != 200:
                print("❌ File upload failed:", r.text)
                continue

            uploaded_file = r.json().get("file_path")
            print(f"✅ File uploaded successfully: {uploaded_file}")
            print("Now, what do you want to ask about this file?")
            msg = input("You: ").strip()

            if not msg:
                print("⚠️ No question provided.")
                continue

            payload = {"session_id": session_id, "message": msg, "filepath": uploaded_file}

        else:
            payload = {"session_id": session_id, "message": msg}

        try:
            r = requests.post(f"{BASE_URL}/chat", json=payload)
            if r.status_code == 200:
                print("AI:", r.json().get("response", "[No response]"))
            else:
                print("❌ Error:", r.text)
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")

def delete_session(username):
    """
    Delete a chat session by selecting from the list.
    """
    sessions = list_sessions(username)
    if not sessions:
        print("⚠️ No sessions to delete.")
        return

    try:
        num = int(input("\nEnter session number to delete: "))
        selected = sessions[num - 1]
    except (ValueError, IndexError):
        print("❌ Invalid session number.")
        return

    confirm = input(f"Are you sure you want to delete '{selected['title']}'? (y/n): ").strip().lower()
    if confirm != "y":
        print("🚫 Deletion canceled.")
        return

    r = requests.delete(f"{BASE_URL}/sessions/{selected['session_id']}")
    if r.status_code == 200:
        print(f"🗑️ Session '{selected['title']}' deleted successfully.")
    else:
        print("❌ Error deleting session:", r.text)

# ---------------- MAIN MENU ----------------
def main_menu(username):
    """
    Show main menu in a loop until user chooses to quit.
    """
    while True:
        sessions = list_sessions(username)

        print("\n=== Main Menu ===")
        print("1. Select existing chat session")
        print("2. Create new chat session")
        print("3. Delete a session")
        print("4. Logout / Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            if not sessions:
                print("⚠️ No sessions available. Please create one first.")
                continue
            try:
                num = int(input("Enter session number: "))
                selected = sessions[num - 1]
                print(f"✅ Selected session: {selected['title']} ({selected['session_id']})")
                chat_loop(selected["session_id"])
            except (ValueError, IndexError):
                print("❌ Invalid session number.")
        elif choice == "2":
            s = create_session(username)
            if s:
                chat_loop(s["session_id"])
        elif choice == "3":
            delete_session(username)
        elif choice == "4":
            print("👋 Logged out. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please enter 1–4.")


# ---------------- ENTRYPOINT ----------------
def main():
    print("=== CLI Chat Application ===\n")
    username = login()
    main_menu(username)


if __name__ == "__main__":
    main()

