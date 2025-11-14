from db_models.models import SessionLocal
from db_models import crud_operations as crud

def seed_users():
    db = SessionLocal()
    try:
        existing = db.query(crud.User).count()
        if existing > 0:
            print("DB already seeded (users exist).")
            return
        print("Seeding users...")
        crud.create_user(db, "shruthi", "pass123")
        crud.create_user(db, "bob", "bobpass")
        crud.create_user(db, "admin", "adminpass")
        print("Seed complete.")
    finally:
        db.close()

if __name__ == "__main__":
    crud.ensure_db()
    seed_users()