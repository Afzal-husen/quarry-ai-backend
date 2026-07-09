import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.core.paths import get_data_dir

# Resolve database path under the configured data directory
DB_DIR = get_data_dir()
DB_PATH = DB_DIR / "users.db"


class UserDatabaseManager:
    """Manages raw SQLite database connections and user operations."""

    @classmethod
    def initialize_db(cls) -> None:
        """Creates the data/ directory and initializes all tables on startup."""
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
            """)
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def get_connection(cls) -> sqlite3.Connection:
        """Returns a standard SQLite connection with foreign keys and WAL enabled."""
        conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    @classmethod
    def get_user_by_username(cls, username: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user record by username.

        Args:
            username: The unique username to look up.

        Returns:
            A dictionary of user fields, or None if the user does not exist.
        """
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, username, hashed_password FROM users WHERE username = ?;",
                (username,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @classmethod
    def get_user_by_id(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user record by user_id."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, username, hashed_password FROM users WHERE id = ?;",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @classmethod
    def create_user(cls, user_id: str, username: str, hashed_password: str) -> Dict[str, Any]:
        """Registers a new user inside the SQLite database.

        Args:
            user_id: Unique UUID to assign to the new user.
            username: Display username (must be unique).
            hashed_password: Hashed password using bcrypt.

        Returns:
            A dictionary representing the created user database record.

        Raises:
            ValueError: If the username already exists.
        """
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (id, username, hashed_password) VALUES (?, ?, ?);",
                (user_id, username, hashed_password)
            )
            conn.commit()
            return {
                "id": user_id,
                "username": username,
                "hashed_password": hashed_password
            }
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: users.username" in str(e):
                raise ValueError(
                    f"Username '{username}' already exists.") from e
            raise e
        finally:
            conn.close()

    @classmethod
    def create_refresh_token(cls, token: str, user_id: str, expires_at: str) -> None:
        """Stores a cryptographically secure refresh token linked to a user."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO refresh_tokens (token, user_id, expires_at) VALUES (?, ?, ?);",
                (token, user_id, expires_at)
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def get_refresh_token(cls, token: str) -> Optional[Dict[str, Any]]:
        """Retrieves a refresh token record."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT token, user_id, expires_at, created_at FROM refresh_tokens WHERE token = ?;",
                (token,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @classmethod
    def delete_refresh_token(cls, token: str) -> None:
        """Deletes/revokes a refresh token from the database."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM refresh_tokens WHERE token = ?;",
                (token,)
            )
            conn.commit()
        finally:
            conn.close()


class ChatDatabaseManager:
    """Manages raw SQLite database connections and chat session/message operations."""

    @classmethod
    def get_connection(cls) -> sqlite3.Connection:
        """Returns a standard SQLite connection with foreign keys enabled."""
        return UserDatabaseManager.get_connection()

    @classmethod
    def create_session(cls, session_id: str, user_id: str, title: str) -> Dict[str, Any]:
        """Creates a new chat session."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?);",
                (session_id, user_id, title)
            )
            conn.commit()
            return {
                "id": session_id,
                "user_id": user_id,
                "title": title
            }
        finally:
            conn.close()

    @classmethod
    def get_session(cls, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a chat session by id, verifying user ownership."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, user_id, title, created_at FROM chat_sessions WHERE id = ? AND user_id = ?;",
                (session_id, user_id)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    @classmethod
    def get_session_by_id(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a chat session by id without verifying user ownership."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, user_id, title, created_at FROM chat_sessions WHERE id = ?;",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


    @classmethod
    def list_sessions(cls, user_id: str, limit: int, offset: int) -> List[Dict[str, Any]]:
        """Lists chat sessions for a specific user, ordered by creation date descending (newest first)."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, user_id, title, created_at FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?;",
                (user_id, limit, offset)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def count_sessions(cls, user_id: str) -> int:
        """Returns the total number of chat sessions for a user."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM chat_sessions WHERE user_id = ?;",
                (user_id,)
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    @classmethod
    def delete_session(cls, session_id: str, user_id: str) -> bool:
        """Deletes a chat session, verifying ownership. Cascades deletion to messages."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?;",
                (session_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    @classmethod
    def update_session_title(cls, session_id: str, user_id: str, title: str) -> bool:
        """Updates the title of a chat session, verifying ownership."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ? AND user_id = ?;",
                (title, session_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    @classmethod
    def create_message(cls, message_id: str, session_id: str, role: str, content: str, metadata: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new message under a chat session."""
        conn = cls.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO chat_messages (id, session_id, role, content, metadata) VALUES (?, ?, ?, ?, ?);",
                (message_id, session_id, role, content, metadata)
            )
            conn.commit()
            return {
                "id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata
            }
        finally:
            conn.close()

    @classmethod
    def get_messages_by_session(cls, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological messages for a session after checking user ownership."""
        conn = cls.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            # Check session ownership first
            cursor.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?;",
                (session_id, user_id)
            )
            if not cursor.fetchone():
                return []

            cursor.execute(
                "SELECT id, session_id, role, content, metadata, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC;",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
