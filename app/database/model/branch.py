from sqlalchemy import Index, Integer, String, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional

from app.database.model.base_model import Base


class Branch(Base):
    """Master table untuk cabang/region"""
    __tablename__ = 'branch'
    __table_args__ = (
        PrimaryKeyConstraint('branch_sid', name='branch_pkey'),
        Index('ix_branch_code', 'code'),
        Index('ix_branch_name', 'name'),
        {'comment': 'Master table untuk cabang/region. Setiap cabang mengelola beberapa toko. Data range: Oct 2024 - Sep 2025.'}
    )

    branch_sid: Mapped[int] = mapped_column(Integer, primary_key=True, comment='Primary key - Unique identifier untuk cabang')
    code: Mapped[Optional[str]] = mapped_column(String(20), comment='Kode cabang (e.g., GJKT TBSD) - Unique identifier untuk business operations')
    name: Mapped[Optional[str]] = mapped_column(String(50), comment='Nama lengkap cabang (e.g., CENTRAL KITCHEN ANCOL, FLAGSHIP BSD)')
