"""Assign a role to an existing local account using operator database access.

Register the account normally, then run:
    python scripts/set_user_role.py --email student@example.org --role admin
No default administrator/password is created or exposed by the web application.
"""
import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.postgres import async_session_maker
from app.models.user import RoleEnum, User


async def assign_role(email: str, role: RoleEnum) -> None:
    async with async_session_maker() as db:
        user = await db.scalar(select(User).where(User.email == email.strip()))
        if user is None:
            raise SystemExit("Account not found. Register the account first.")
        if not user.is_active:
            raise SystemExit("Account is inactive; role changes will not reactivate it.")
        user.role = role
        await db.commit()
    print(f"Assigned role {role.value} to the existing account.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", required=True, choices=[role.value for role in RoleEnum])
    args = parser.parse_args()
    asyncio.run(assign_role(args.email, RoleEnum(args.role)))


if __name__ == "__main__":
    main()
