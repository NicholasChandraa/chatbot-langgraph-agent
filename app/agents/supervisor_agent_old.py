from langgraph_supervisor import create_supervisor
# Library agar supervisor bisa menyerahkan response dari sub agents secara langsung tanpa dikembalikan ke supervisor agent lalu di re-process
from langgraph_supervisor.handoff import create_forward_message_tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.sub_agent.product_agent import create_product_agent
from app.agents.sub_agent.sales_agent import create_sales_agent
from app.agents.sub_agent.store_agent import create_store_agent

from app.agents.tools.custom_handoff_tool import create_custom_handoff_tool

from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_supervisor_agent(db: AsyncSession):
    """
    Create supervisor that manages specialized business intelligence agents.

    The supervisor routes questions to appropriate agents based on content:
    - Product questions -> Product Agent
    - Sales questions -> Sales Agent
    - Store questions -> Store Agent

    Args:
        db: Database session for loading agent configs
    
    Returns:
        Compiled supervisor graph ready to process queries
    """

    logger.info("☑️ Creating Supervisor Agent...")


    # Create worker agents
    product_agent = await create_product_agent(db)
    sales_agent = await create_sales_agent(db)
    store_agent = await create_store_agent(db)

    # Load supervisor config
    config = await get_agent_config("supervisor", db)

    # Create supervisor LLM
    supervisor_llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create custom handoff tools with task description support
    logger.info("🔧 Creating handoff tools for supervisor...")

    product_handoff = create_custom_handoff_tool(
        agent_name="product_agent",
        name="assign_to_product_agent",
        description=(
            "Assign product-related tasks to the product specialist. "
            "Use for: product information, pricing, PLU codes, product categories, availability. "
        )
    )
    logger.info("✅ product_handoff (assign_to_product_agent) created")

    sales_handoff = create_custom_handoff_tool(
        agent_name="sales_agent",
        name="assign_to_sales_agent",
        description=(
            "Assign sales analytics tasks to the sales specialist. "
            "Use for: sales revenue, trends, top products, performance analysis, sales reports."
        )
    )
    logger.info("✅ sales_handoff (assign_to_sales_agent) created")

    store_handoff = create_custom_handoff_tool(
        agent_name="store_agent",
        name="assign_to_store_agent",
        description=(
            "Assign store information tasks to the store specialist."
            "Use for: store details, branch information, locations, store counts."
        )
    )
    logger.info("✅ store_handoff (assign_to_store_agent) created")

    # Create forwarding tool (note: we'll track its usage through supervisor behavior instead)
    logger.info("🔧 Creating forwarding_tool...")
    forwarding_tool = create_forward_message_tool("supervisor")
    logger.info("✅ forwarding_tool created")
    logger.info(f"📋 forwarding_tool details: name={forwarding_tool.name}, description={forwarding_tool.description}")

    # Create supervisor using built-in function
    supervisor_graph = create_supervisor(
        model=supervisor_llm,
        agents=[product_agent, sales_agent, store_agent],
        tools=[product_handoff, sales_handoff, store_handoff, forwarding_tool],
        prompt=(
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

            "ROUTING LOGIC:\n"
            "- Untuk pertanyaan produk → assign_to_product_agent\n"
            "- Untuk pertanyaan penjualan → assign_to_sales_agent\n"
            "- Untuk pertanyaan toko → assign_to_store_agent\n"
            "- Untuk pertanyaan meta/greetings → jawab LANGSUNG sendiri\n\n"

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
        ),
        add_handoff_back_messages=False,
        output_mode="full_history",
        supervisor_name="supervisor_agent",
        add_handoff_messages=True,
        parallel_tool_calls=False,
        include_agent_name=None
    )

    logger.info("✅ Supervisor Agent Created (uncompiled)")

    return supervisor_graph