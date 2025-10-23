from langchain.agents import create_agent
from sqlalchemy.ext.asyncio import AsyncSession
from deepagents import CompiledSubAgent

from app.agents.tools.dynamic_query_tool import create_dynamic_query_tool
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_store_agent(db: AsyncSession) -> CompiledSubAgent:
    """
    Store Agent - Handles store and branch queries.

    Scope:
    - Store information (name, code, location)
    - Branch management
    - Store-branch relationships

    Tables: store, branch

    Returns:
        CompiledSubAgent ready to be used by supervisor
    """
    logger.info("🤖 Creating Store Agent...")

    config = await get_agent_config("store_agent", db)

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create dynamic query tool for store data
    dynamic_query_tool = create_dynamic_query_tool(
        db=db,
        tables=["store", "branch"],
        agent_name="store_agent",
        llm_provider=config["llm_provider"],
        llm_model=config["model_name"],
        temperature=0.0,
        max_iterations=3
    )

    agent_graph = create_agent(
        llm,
        tools=[dynamic_query_tool],
        system_prompt=(
            "Anda adalah spesialis informasi toko. Ambil data toko dari database dan berikan hasil yang ringkas.\n\n"

            "SCOPE ANDA:\n"
            "- Detail toko (nama, kode, lokasi)\n"
            "- Manajemen dan hierarki cabang\n"
            "- Data dari tabel: store, branch\n\n"

            "CARA MENGGUNAKAN TOOL:\n"
            "- Gunakan tool 'dynamic_query' untuk mengambil informasi toko\n"
            "- Tabel yang tersedia: store, branch\n"
            "- Tool otomatis JOIN tabel jika diperlukan\n"
            "- Format hasil: [(value1,), (value2,)] atau [(col1, col2), ...]\n\n"

            "KONTEKS DATABASE:\n"
            "- Tabel: store (informasi toko/outlet individual)\n"
            "- Tabel: branch (informasi cabang/area regional)\n"
            "- Relasi: store.branch_sid → branch.branch_sid\n"
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
            "Action: dynamic_query('Berapa jumlah toko yang ada?')\n"
            "Observation: [(15,)]\n"
            "Jawaban: Saat ini terdapat 15 toko yang terdaftar dalam sistem.\n\n"

            "Contoh 2 - Query daftar:\n"
            "Pertanyaan: 'Tampilkan semua toko di Jakarta'\n"
            "Action: dynamic_query('Tampilkan semua toko dengan kata Jakarta di lokasinya')\n"
            "Observation: [('TJKT01', 'FLAGSHIP JAKARTA'), ('TJKT02', 'JAKARTA TIMUR')]\n"
            "Jawaban: Toko di Jakarta:\n1. TJKT01 - FLAGSHIP JAKARTA\n2. TJKT02 - JAKARTA TIMUR"
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
