import datetime
from sqlalchemy import DateTime, Index, Integer, Numeric, String, ForeignKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional

from app.database.model.base_model import Base




class StoreDailySingleItem(Base):
    """Fact table - Daily sales transactions per product per store"""
    __tablename__ = 'store_daily_single_item'
    __table_args__ = (
        ForeignKeyConstraint(['product_sid'], ['product.product_sid'], ondelete='RESTRICT', onupdate='CASCADE', name='fk_store_daily_single_item_product'),
        ForeignKeyConstraint(['store_sid'], ['store.store_sid'], ondelete='RESTRICT', onupdate='CASCADE', name='fk_store_daily_single_item_store'),
        Index('idx_sdsi_date', 'date'),
        Index('ix_store_daily_single_item_date', 'date'),
        Index('ix_store_daily_single_item_product_sid', 'product_sid'),
        Index('ix_store_daily_single_item_store_sid', 'store_sid'),
        {'comment': 'Fact table - Daily sales transactions per product per store. Main table for sales analytics. Data range: Oct 2024 - Sep 2025 (365 days). Contains both transaction-level and aggregated data.'}
    )

    # Composite Primary Key (product_sid, store_sid, date, sales_code)
    product_sid: Mapped[int] = mapped_column(Integer, primary_key=True, comment='Part of composite primary key - Foreign key ke product.product_sid. ALWAYS JOIN with product table to get product details.')
    store_sid: Mapped[int] = mapped_column(Integer, primary_key=True, comment='Part of composite primary key - Foreign key ke store.store_sid. ALWAYS JOIN with store table to get store name/code.')
    date: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True, comment='Part of composite primary key - Transaction/aggregation date. ALWAYS use CAST(date AS DATE) for date comparisons. Format: TIMESTAMP (e.g., 2025-01-15 00:00:00)')
    sales_code: Mapped[str] = mapped_column(String(10), primary_key=True, comment="Kode transaksi/receipt number. Can be NULL or empty string for aggregated records. Use COUNT(DISTINCT NULLIF(sales_code, '')) for counting unique transactions.")
    
    # Measures
    qty_sales: Mapped[int] = mapped_column(Integer, comment='Quantity terjual dalam satuan unit/pieces. Can be 0 for days with no sales. Use filter (qty_sales > 0 OR rp_sales > 0) for valid sales data.')
    rp_sales: Mapped[Optional[float]] = mapped_column(Numeric(20, 5), comment='Nominal penjualan dalam Rupiah. Precision: NUMERIC(20,5). Can be 0 for days with no sales. Use filter (qty_sales > 0 OR rp_sales > 0) for valid sales data.')