import os

from deepagents import create_deep_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.sub_agent.product_agent import create_product_agent
from app.agents.sub_agent.sales_agent import create_sales_agent
from app.agents.sub_agent.store_agent import create_store_agent


from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_supervisor_agent(db: AsyncSession, checkpointer=None, store=None, user_context: str = ""):
    """
    Create supervisor that manages specialized business intelligence agents using DeepAgents.

    The supervisor delegates tasks to appropriate subagents based on content:
    - Product questions -> product_agent
    - Sales questions -> sales_agent
    - Store questions -> store_agent

    Args:
        db: Database session for loading agent configs
        checkpointer: Optional checkpointer for persistent memory
        store: Optional store for long-term memory
        user_context: User context string from long-term memory

    Returns:
        Compiled deep agent ready to process queries
    """

    logger.info("☑️ Creating Supervisor Agent with DeepAgents...")

    # Load configs for all agents
    llm_config = await get_agent_config("supervisor", db)

    # Create supervisor LLM
    supervisor_llm = LLMProviderFactory.create_from_config(llm_config)

    logger.info(f"✅ Supervisor LLM created: {llm_config['model_name']}")
    
    # panggil sub agent
    product_agent = await create_product_agent(db)
    sales_agent = await create_sales_agent(db)
    store_agent = await create_store_agent(db)

    subagents = [product_agent, sales_agent, store_agent]


    # Define subagents as dictionaries (DeepAgents pattern)
    logger.info("📦 Defining subagents...")

    # Build base system prompt
    base_system_prompt = (
        "Anda adalah Asisten Virtual Mister Donut yang ramah dan membantu.\n\n"

        "IDENTITAS ANDA:\n"
        "- Nama: Asisten Virtual Mister Donut\n"
        "- Peran: Membantu customer dengan informasi produk, penjualan, dan toko\n"
        "- Kepribadian: Ramah, helpful, profesional\n\n"

        "KEMAMPUAN ANDA:\n"
        "Anda bisa membantu customer dengan:\n"
        "- Informasi produk (harga, PLU code, kategori)\n"
        "- Data penjualan dan analisis (revenue, produk terlaris, trend)\n"
        "- Informasi toko (lokasi, cabang, jumlah outlet)\n\n"

        "CARA KERJA INTERNAL (RAHASIA - JANGAN EXPOSED KE USER):\n"
        "Anda punya 3 specialized agents yang bekerja di background:\n"
        "1. product_agent - untuk pertanyaan produk\n"
        "2. sales_agent - untuk pertanyaan penjualan\n"
        "3. store_agent - untuk pertanyaan toko\n\n"

        "ATURAN PENTING:\n"
        "1. JANGAN pernah reveal bahwa Anda punya sub-agents\n"
        "2. JANGAN bilang 'saya akan routing ke agent lain'\n"
        "3. Ketika agent memberikan jawaban, gunakan 'forward_message' untuk kirim langsung\n"
        "4. JANGAN tambahkan commentary setelah agent selesai\n"
        "5. Present semua jawaban seolah-olah ANDA yang menjawab\n\n"

        "HANDLING META QUESTIONS:\n"
        "Jika user tanya 'Siapa kamu?' atau 'Apa yang bisa kamu lakukan?':\n"
        "- Jawab: 'Saya adalah Asisten Virtual Mister Donut yang siap membantu Anda dengan informasi produk, penjualan, dan toko kami.'\n"
        "- JANGAN mention sub-agents atau routing mechanism\n\n"

        "MULTI-PART QUESTIONS:\n"
        "Jika pertanyaan punya meta + data (contoh: 'Siapa kamu? Berapa harga donut?'):\n"
        "1. Jawab meta part dengan identitas Anda\n"
        "2. Route data part ke agent yang sesuai\n"
        "3. Gabungkan jawaban jadi satu response cohesive\n\n"

        "BAHASA:\n"
        "- Gunakan bahasa yang sama dengan user (Indonesia/English)\n"
        "- Tone: Friendly, helpful, professional"
    )

    # Inject user context if available
    if user_context:
        system_prompt = (
            f"{base_system_prompt}\n\n"
            f"📋 INFORMASI USER (dari riwayat interaksi sebelumnya):\n"
            f"{user_context}\n\n"
            f"CARA MENGGUNAKAN INFORMASI USER:\n"
            f"- Sapa user dengan nama jika tersedia\n"
            f"- Berikan rekomendasi berdasarkan produk favorit mereka\n"
            f"- Perhatikan pantangan makanan (dietary restrictions)\n"
            f"- Gunakan informasi ini untuk memberikan respons yang lebih personal dan relevan\n"
            f"- JANGAN reveal bahwa Anda punya 'database' atau 'system' - buat natural seperti Anda 'mengingat' mereka"
        )
        logger.info("✅ User context injected to supervisor prompt")
    else:
        system_prompt = base_system_prompt
        logger.debug("No user context to inject")

    supervisor_graph = create_deep_agent(
        model=supervisor_llm,
        subagents=subagents,
        checkpointer=checkpointer,
        store=store,
        system_prompt=system_prompt,
    )

    memory_status = "with checkpointer" if checkpointer else "without checkpointer"
    logger.info(f"✅ Supervisor Agent Created ({memory_status})")

    return supervisor_graph