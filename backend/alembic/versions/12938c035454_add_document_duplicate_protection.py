
"""add document duplicate protection

Revision ID: 12938c035454
Revises: 039492c62994
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "12938c035454"
down_revision: Union[str, Sequence[str], None] = "039492c62994"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "duplicate_protection_enabled",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )

    op.create_index(
        "uq_documents_user_content_hash_protected",
        "documents",
        ["user_id", "content_hash"],
        unique=True,
        postgresql_where=sa.text(
            "duplicate_protection_enabled IS TRUE "
            "AND content_hash IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_documents_user_content_hash_protected",
        table_name="documents",
    )
    op.drop_column("documents", "duplicate_protection_enabled")
