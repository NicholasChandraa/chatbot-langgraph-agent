from langchain_core.tools import tool

from app.utils.logger import logger
from app.repositories.product_repository import ProductRepository

# Define tool using @tool decorator
@tool
async def product_query(question: str, repo: ProductRepository) -> str:
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