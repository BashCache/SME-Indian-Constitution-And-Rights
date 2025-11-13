import os, json, uuid
from datetime import datetime

BASE_DIR = "agent_data"
USERS_FILE = os.path.join(BASE_DIR, "users.json")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({"shruthi": "pass123", "demo": "demo123"}, f, indent=2)

def validate_user(username: str, password: str) -> bool:
    with open(USERS_FILE) as f:
        users = json.load(f)
    return username in users and users[username] == password


# ---------- Session Operations ----------
def list_sessions(username: str):
    sessions = []
    for file in os.listdir(SESSIONS_DIR):
        if file.startswith(username):
            path = os.path.join(SESSIONS_DIR, file)
            with open(path) as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "title": data["title"],
                    "created_at": data["created_at"]
                })
    return sessions


def create_session(username: str, title: str):
    sid = f"{username}_{uuid.uuid4().hex[:6]}"
    session = {
        "session_id": sid,
        "username": username,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    with open(os.path.join(SESSIONS_DIR, f"{sid}.json"), "w") as f:
        json.dump(session, f, indent=2)
    return session


def get_session(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
