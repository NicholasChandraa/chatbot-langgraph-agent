from sqlalchemy import Index, Integer, String, PrimaryKeyConstraint, ForeignKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from app.database.model.base_model import Base


class Store(Base):
    """Master table untuk toko/outlet"""
    __tablename__ = 'store_master'
    __table_args__ = (
        ForeignKeyConstraint(['branch_sid'], ['branch.branch_sid'], name='store_branch_sid_fkey'),
        PrimaryKeyConstraint('store_sid', name='store_pkey'),
        Index('ix_store_branch_sid', 'branch_sid'),
        Index('ix_store_code', 'code'),
        Index('ix_store_name', 'name'),
        {'comment': 'Master table untuk toko/outlet. Setiap toko belongs to satu cabang. Contains flagship stores dan regular outlets.'}
    )

    store_sid: Mapped[int] = mapped_column(Integer, primary_key=True, comment='Primary key - Unique identifier untuk toko')
    code: Mapped[Optional[str]] = mapped_column(String(20), comment='Kode toko (e.g., TLPC, TCWS, TPLG) - Unique identifier untuk business operations. Case-sensitive untuk exact match.')
    name: Mapped[Optional[str]] = mapped_column(String(50), comment='Nama lengkap toko (e.g., FS LIPPO CIKARANG, FLAGSHIP CIPUTRA WORLD) - Use ILIKE for search')
    branch_sid: Mapped[Optional[int]] = mapped_column(Integer, comment='Foreign key - branch.branch_sid - Cabang yang mengelola toko ini')

