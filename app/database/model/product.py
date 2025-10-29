# coding: utf-8
# Using sqlacodegen - Simplified for Agent Service
# Only includes tables used by AI: branch, store_master, product, store_daily_single_item
# Nanti ini bisa dipecah 1 table 1 file, untuk kedepannya

from sqlalchemy import Index, Integer, String, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from app.database.model.base_model import Base

class Product(Base):
    """Master table untuk produk"""
    __tablename__ = 'product'
    __table_args__ = (
        PrimaryKeyConstraint('product_sid', name='product_pkey'),
        Index('ix_product_plu', 'plu'),
        Index('ix_product_plu_name', 'plu_name'),
        Index('ix_product_product_sid', 'product_sid'),
        Index('product_name_lower_idx'),
        Index('product_name_trigram_idx'),
        Index('product_search_vector_idx', 'search_vector'),
        {'comment': 'Master table untuk produk. Contains all products sold across stores including bakery items, beverages, and specialty products.'}
    )

    plu: Mapped[str] = mapped_column(String(20), comment='Product Lookup Code - Unique business identifier. String with leading zeros (e.g., 01040109, 00000220)')
    product_sid: Mapped[int] = mapped_column(Integer, primary_key=True, comment='Primary key - Unique identifier untuk produk')
    plu_name: Mapped[Optional[str]] = mapped_column(String(50), comment='Nama produk (e.g., COOKIES RING - FS, ICED CHOCOLATE (REG), MOCHIDO DOUBLE CHOCO - FS). Use ILIKE with wildcards for search.')
    search_vector: Mapped[Optional[any]] = mapped_column(TSVECTOR, comment='Full-text search vector (auto-generated) - For advanced text search capabilities. Can be ignored for basic queries.')


