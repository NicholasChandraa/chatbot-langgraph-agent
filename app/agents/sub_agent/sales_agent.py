from langchain.agents import create_agent
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.sales_repository import SalesRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_sales_agent(repo: SalesRepository) -> CompiledSubAgent:
    """
    Sales Agent - Handles sales analytics queries.

    Scope:
    - Sales revenue and trends
    - Top selling products
    - Store performance
    - Sales by date/period

    Tables: store_daily_single_item, product, store_master, branch

    Args:
        repo: SalesRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready for supervisor
    """
    logger.info("🤖 Creating Sales Agent...")

    config = await repo.get_config()

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool with @tool decorator
    @tool("sales_dynamic_query")
    async def sales_query(question: str) -> str:
        """
        Query sales analytics database using natural language

        This tool automatically:
        1. Converts your natural langauge question to SQL
        2. Handles complex joins across sales, product, store, branch tables
        3. Executes the query safely
        4. Returns formatted results

        Use this tool to get sales analytics like:
        - Total revenue by date/period
        - Top selling products
        - Store performance comparison
        - Sales trends and patterns

        Args:
            question (str): Natural language question about sales

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Berapa total penjualan kemarin?"
            "Tampilkan 5 produk terlaris minggu ini"
            "Bandingkan performa toko FS Palembang vs FS Bandung"
        """
        try:
            logger.info(f"[sales_agent] Tool called: {question[:50]}...")

            # Execute query via repository
            result = await repo.execute_query(question)

            logger.info(f"[sales_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying sales data: {str(e)}"
            logger.error(f"[sales_agent] {error_msg}", exc_info=True)
            return error_msg

    agent_graph = create_agent(
        llm,
        tools=[sales_query],
        system_prompt=(
            "Anda adalah spesialis analisis penjualan. Ambil data penjualan dari database dan berikan hasil analisis yang ringkas.\n\n"

            "SCOPE ANDA:\n"
            "- Analisis revenue dan trend penjualan\n"
            "- Identifikasi produk terlaris\n"
            "- Perbandingan performa toko\n"
            "- Data dari tabel: store_daily_single_item, product, store, branch\n\n"

            "CARA MENGGUNAKAN TOOL:\n"
            "- Gunakan tool 'sales_dynamic_query' untuk mengambil data penjualan\n"
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
            "Action: sales_dynamic_query('Berapa total revenue untuk hari ini?')\n"
            "Observation: [(2500000,)]\n"
            "Jawaban: Total penjualan hari ini mencapai Rp 2.500.000\n\n"

            "Contoh 2 - Query produk terlaris:\n"
            "Pertanyaan: 'Produk apa yang paling laris minggu ini?'\n"
            "Action: sales_dynamic_query('Tampilkan 5 produk terlaris minggu ini berdasarkan quantity dengan nama produk')\n"
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