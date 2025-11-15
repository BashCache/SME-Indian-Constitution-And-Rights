# memory_store.py

# Structure:
# {
#    session_id: [
#        {"role": "user", "content": "..."},
#        {"role": "assistant", "content": "..."}
#    ]
# }

MEMORY: dict[str, list] = {}


def get_memory(session_id: str):
    return MEMORY.get(session_id, [])


def append_to_memory(session_id: str, role: str, content: str):
    if session_id not in MEMORY:
        MEMORY[session_id] = []
    MEMORY[session_id].append({"role": role, "content": content})
