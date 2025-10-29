from langchain.agents import create_agent
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.store_repository import StoreRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_store_agent(repo: StoreRepository) -> CompiledSubAgent:
    """
    Store Agent - Handles store and branch queries.

    Scope:
    - Store information (name, code, location)
    - Branch management
    - Store-branch relationships

    Tables: store_master, branch

    Args:
        repo: StoreRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready for supervisor
    """
    logger.info("🤖 Creating Store Agent...")

    config = await repo.get_config()

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool with @tool decorator
    @tool("store_dynamic_query")
    async def store_query(question: str) -> str:
        """
        Query store database using natural language

        This tool automatically:
        1. Converts your natural language question to SQL
        2. Handles joins across store and branch tables
        3. Executes the query safely
        4. Returns formatted results

        Use this tool to get store information like:
        - Store locations and addresses
        - Branch information
        - Store status and availability

        Args:
            question (str): Natural language question about stores

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Berapa jumlah toko yang ada"
            "Cari toko dengan kode TPLG"
        """
        try:
            logger.info(f"[store_agent] Tool called: {question[:50]}")

            # Execute query via repository
            result = await repo.execute_query(question)

            logger.info(f"[store_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying store data: {str(e)}"
            logger.error(f"[store_agent] {error_msg}", exc_info=True)
            return error_msg

    agent_graph = create_agent(
        llm,
        tools=[store_query],
        system_prompt=(
            "Anda adalah spesialis informasi toko. Ambil data toko dari database dan berikan hasil yang ringkas.\n\n"

            "SCOPE ANDA:\n"
            "- Detail toko (nama, kode, lokasi)\n"
            "- Manajemen dan hierarki cabang\n"
            "- Data dari tabel: store_master, branch\n\n"

            "CARA MENGGUNAKAN TOOL:\n"
            "- Gunakan tool 'store_dynamic_query' untuk mengambil informasi toko\n"
            "- Tabel yang tersedia: store_master, branch\n"
            "- Tool otomatis JOIN tabel jika diperlukan\n"
            "- Format hasil: [(value1,), (value2,)] atau [(col1, col2), ...]\n\n"

            "KONTEKS DATABASE:\n"
            "- Tabel: store_master (informasi toko/outlet individual)\n"
            "- Tabel: branch (informasi cabang/area regional)\n"
            "- Relasi: store_master.branch_sid → branch.branch_sid\n"
            "- Kode toko case-sensitive (contoh: 'TLPC', 'TCWS', 'TPLG')\n"
            "- Nama toko dalam HURUF BESAR\n\n"

            "FORMAT RESPONSE:\n"
            "- Sertakan kode toko dan nama lengkap\n"
            "- Sebutkan cabang/area jika relevan\n"
            "- Ringkas dan informatif (maksimal 500 kata)\n"
            "- Gunakan bahasa yang sama dengan user (Indonesia/English)\n"
            "- Jika hasil kosong [], katakan 'Tidak ada data ditemukan'\n\n"

            "HANDLING DATA TIDAK TERSEDIA:\n"
            "- Jika user tanya informasi yang tidak ada di schema (contoh: jam buka, nomor kontak):\n"
            "  * Akui pertanyaannya\n"
            "  * Jelaskan dengan sopan bahwa informasi spesifik tidak tersedia\n"
            "  * Tawarkan informasi alternatif yang BISA Anda berikan (nama, kode, lokasi, cabang)\n"
            "- Contoh: 'Maaf, informasi jam operasional tidak tersedia. Namun saya bisa memberikan informasi nama toko, kode, dan lokasinya.'\n\n"

            "CONTOH:\n\n"

            "Contoh 1 - Query jumlah:\n"
            "Pertanyaan: 'Berapa jumlah toko yang aktif?'\n"
            "Action: store_dynamic_query('Berapa jumlah toko yang ada?')\n"
            "Observation: [(15,)]\n"
            "Jawaban: Saat ini terdapat 15 toko yang terdaftar dalam sistem.\n\n"

            "Contoh 2 - Query daftar:\n"
            "Pertanyaan: 'Tampilkan semua toko di Jakarta'\n"
            "Action: store_dynamic_query('Tampilkan toko dengan kode TPLG')\n"
            "Observation: [('TPLG', 'FS PALEMBANG'),]\n"
            "Jawaban: Toko dengan kode TPLG adalah toko FS Palembang"
        ),
        name="store_agent"
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="store_agent",
        description=(
            "Store information specialist for handling store and branch queries. "
            "Use for: store details, branch information, locations, store counts."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Store Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent
