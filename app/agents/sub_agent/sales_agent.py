from langchain.agents import create_agent
from sqlalchemy.ext.asyncio import AsyncSession
from deepagents import CompiledSubAgent

from app.agents.tools.dynamic_query_tool import create_dynamic_query_tool
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_sales_agent(db: AsyncSession) -> CompiledSubAgent:
    """
    Sales Agent - Handles sales analytics queries.

    Scope:
    - Sales revenue and trends
    - Top selling products
    - Store performance
    - Sales by date/period

    Tables: store_daily_single_item, product, store, branch

    Returns:
        CompiledSubAgent ready to be used by supervisor
    """
    logger.info("🤖 Creating Sales Agent...")

    config = await get_agent_config("sales_agent", db)

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create dynamic query tool (with multiple tables for analytics)
    dynamic_query_tool = create_dynamic_query_tool(
        db=db,
        tables=["store_daily_single_item", "product", "store", "branch"],
        agent_name="sales_agent",
        llm_provider=config["llm_provider"],
        llm_model=config["model_name"],
        temperature=0.0,
        max_iterations=3
    )

    agent_graph = create_agent(
        llm,
        tools=[dynamic_query_tool],
        system_prompt=(
            "Anda adalah spesialis analisis penjualan. Ambil data penjualan dari database dan berikan hasil analisis yang ringkas.\n\n"

            "SCOPE ANDA:\n"
            "- Analisis revenue dan trend penjualan\n"
            "- Identifikasi produk terlaris\n"
            "- Perbandingan performa toko\n"
            "- Data dari tabel: store_daily_single_item, product, store, branch\n\n"

            "CARA MENGGUNAKAN TOOL:\n"
            "- Gunakan tool 'dynamic_query' untuk mengambil data penjualan\n"
            "- Tool memiliki akses ke: store_daily_single_item, product, store, branch\n"
            "- Tool otomatis generate SQL optimal dengan JOIN\n"
            "- Tanyakan dalam bahasa natural\n"
            "- Format hasil: [(value1,), (value2,)] atau [(col1, col2, col3), ...]\n\n"

            "KONTEKS DATABASE:\n"
            "- Tabel utama: store_daily_single_item (transaksi penjualan)\n"
            "- Kolom penting:\n"
            "  * qty_sales: Quantity terjual (unit)\n"
            "  * rp_sales: Revenue dalam Rupiah Indonesia\n"
            "  * date: Tanggal transaksi\n"
            "- Tool otomatis JOIN tabel jika diperlukan\n\n"

            "FORMAT RESPONSE:\n"
            "- Format mata uang sebagai: Rp 1.500.000 (dengan pemisah ribuan titik)\n"
            "- Sertakan nama produk/toko, bukan hanya ID\n"
            "- Berikan konteks dan insight, bukan hanya angka mentah\n"
            "- Untuk trend, sebutkan periode waktu dengan jelas\n"
            "- Ringkas dan informatif (maksimal 500 kata)\n"
            "- Gunakan bahasa yang sama dengan user (Indonesia/English)\n"
            "- Jika hasil kosong [], katakan 'Tidak ada data penjualan ditemukan'\n\n"

            "HANDLING DATA TIDAK TERSEDIA:\n"
            "- Jika user tanya metrik yang tidak ada di schema (contoh: profit margin, demografi):\n"
            "  * Akui pertanyaannya\n"
            "  * Jelaskan dengan sopan bahwa metrik spesifik tidak tersedia\n"
            "  * Sarankan metrik alternatif yang BISA Anda berikan (revenue, quantity, trend)\n"
            "- Contoh: 'Maaf, data margin keuntungan tidak tersedia. Namun saya bisa menampilkan total revenue dan quantity penjualan.'\n\n"

            "CONTOH:\n\n"

            "Contoh 1 - Query revenue:\n"
            "Pertanyaan: 'Berapa total penjualan hari ini?'\n"
            "Action: dynamic_query('Berapa total revenue untuk hari ini?')\n"
            "Observation: [(2500000,)]\n"
            "Jawaban: Total penjualan hari ini mencapai Rp 2.500.000\n\n"

            "Contoh 2 - Query produk terlaris:\n"
            "Pertanyaan: 'Produk apa yang paling laris minggu ini?'\n"
            "Action: dynamic_query('Tampilkan 5 produk terlaris minggu ini berdasarkan quantity dengan nama produk')\n"
            "Observation: [('GLAZED DONUT', 250), ('CHOCOLATE GLAZED', 180), ('ICED COFFEE', 150)]\n"
            "Jawaban: Produk terlaris minggu ini:\n1. GLAZED DONUT - 250 unit\n2. CHOCOLATE GLAZED - 180 unit\n3. ICED COFFEE - 150 unit"
        ),
        name="sales_agent"
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="sales_agent",
        description=(
            "Sales analytics specialist for handling sales-related queries. "
            "Use for: sales revenue, trends, top products, performance analysis, sales reports."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Sales Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent