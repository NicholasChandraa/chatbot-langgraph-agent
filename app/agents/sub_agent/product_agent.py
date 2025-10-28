from langchain.agents import create_agent
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.product_repository import ProductRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_product_agent(repo: ProductRepository) -> CompiledSubAgent:
    """
    Product Agent - Handles product-related queries.

    Scope:
    - Product information
    - Product categories
    - Product availability

    Tables: product only

    Args:
        repo: ProductRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready to be used by supervisor
    """
    logger.info("🤖 Creating Product Agent...")

    # Load config from database
    config = await repo.get_config()

    # Create LLM for ReAct agent
    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool using @tool decorator
    @tool
    async def product_query(question: str) -> str:
        """
        Query product database using natural language.

        This tool automatically:
        1. Converts your natural language question to SQL
        2. Executes the query safely
        3. Returns formatted results

        Use this tool to get product information like:
        - Product names, PLU codes
        - Product search by keyword
        - Product availability
    
        Args:
            question (str): Natural language question about products

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Tampilkan semua produk coklat"
            "Cari produk dengan PLU 000000906"
        """
        try:
            logger.info(f"[product_agent] Tool called: {question[:50]}...")

            # Execute query via repository (repository handles caching, metrics, etc.)
            result = await repo.execute_query(question)

            logger.info(f"[product_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying products: {str(e)}"
            logger.error(f"[product_agent] {error_msg}", exc_info=True)
            return error_msg

    # Create React Agent with single tool
    agent_graph = create_agent(
        llm,
        tools=[product_query],
        system_prompt=(
            "Anda adalah spesialis data produk. Ambil informasi produk dari database dan berikan hasil yang ringkas.\n\n"

            "SCOPE ANDA:\n"
            "- Nama produk, harga, kode PLU\n"
            "- Kategori dan varian produk\n"
            "- Data dari tabel 'product' saja\n\n"

            "CARA MENGGUNAKAN TOOL:\n"
            "- Gunakan tool 'product_query' untuk mengambil data produk\n"
            "- Tool menerima pertanyaan dalam bahasa natural\n"
            "- Tool otomatis generate dan eksekusi SQL query\n"
            "- Format hasil: [(value1,), (value2,)] atau [(col1, col2, col3), ...]\n\n"

            "KONTEKS DATABASE:\n"
            "- Nama produk dalam HURUF BESAR (contoh: 'GLAZED DONUT', 'ICED CHOCOLATE (REG)')\n"
            "- Kode PLU adalah string dengan leading zeros (contoh: '01040109', '00000220')\n"
            "- Harga dalam Rupiah Indonesia (IDR) tanpa desimal\n\n"

            "FORMAT RESPONSE:\n"
            "- Format harga sebagai: Rp 15.000 (gunakan pemisah ribuan dengan titik)\n"
            "- Selalu sertakan nama produk dan kode PLU\n"
            "- Ringkas dan informatif (maksimal 500 kata)\n"
            "- Gunakan bahasa yang sama dengan user (Indonesia/English)\n"
            "- Jika hasil kosong [], katakan 'Produk tidak ditemukan'\n\n"

            "HANDLING DATA TIDAK TERSEDIA:\n"
            "- Jika user tanya data yang tidak ada di schema (contoh: stok, expired date):\n"
            "  * Akui pertanyaannya\n"
            "  * Jelaskan dengan sopan bahwa informasi spesifik tidak tersedia\n"
            "  * Tawarkan informasi alternatif yang BISA Anda berikan (harga, kategori)\n"
            "- Contoh: 'Maaf, informasi stok tidak tersedia. Namun saya bisa memberikan informasi harga dan kategori produk.'\n\n"

            "CONTOH:\n\n"

            "Contoh 1 - Query pencarian:\n"
            "Pertanyaan: 'Produk coklat apa saja yang ada?'\n"
            "Action: product_query('Tampilkan semua produk dengan kata chocolate')\n"
            "Observation: [('CHOCOLATE GLAZED', '01040109', 15000), ('ICED CHOCOLATE (REG)', '00000220', 25000)]\n"
            "Jawaban: Produk coklat yang tersedia:\n1. CHOCOLATE GLAZED (PLU: 01040109) - Rp 15.000\n2. ICED CHOCOLATE (REG) (PLU: 00000220) - Rp 25.000"
        ),
        name="product_agent"
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="product_agent",
        description=(
            "Product specialist for handling product-related queries. "
            "Use for: product information, pricing, PLU codes, product categories, availability."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Product Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent
