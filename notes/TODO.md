TODO:

- Untuk bagian-bagian yang sudah di cache kaya di agent_config_manager, itu untuk cache nya mungkin bisa direfactor jadi menggunakan redis. Dikarenakan kalau pake redis bisa lebih visualable.



3 tempat utama yang menggunakan manual in-memory caching:

  ---
  1. AgentConfigManager

  📁 config/agent_config/agent_config_manager.py:15-136

  Cache implementation:
  self._cache: Dict[str, Dict] = {}                # Config data
  self._last_refresh: Dict[str, datetime] = {}     # Timestamp tracking
  self._cache_ttl = 300  # 5 minutes

  Apa yang di-cache:
  - Agent configurations (LLM provider, model, temperature, dll)
  - Per agent: supervisor, product_agent, sales_agent, store_agent, sql_agent

  Cache methods:
  - _is_cache_valid() - Check expiry by timestamp
  - _update_cache() - Update with timestamp
  - invalidate_cache() - Manual invalidation

  TTL: 5 minutes (300 seconds)

  Singleton: agent_config_manager (line 139)

  ---
  2. QueryRepository

  📁 repositories/base/query_repository.py:30-323

  Cache implementation:
  self._cache: Dict[str, Dict[str, Any]] = {}
  self._cache_ttl = 300  # 5 minutes default

  Apa yang di-cache:
  - Natural language query results
  - Cache key: {agent_name}:{hash(question)}
  - Cache value: {"result": str, "timestamp": datetime}

  Cache methods:
  - _get_cache_key() - Generate hash-based key
  - _get_from_cache() - Check expiry + retrieve
  - _add_to_cache() - Add with timestamp + size management (max 100 entries)
  - clear_cache() - Manual clear
  - get_cache_stats() - Stats untuk monitoring

  TTL: 5 minutes (300 seconds, configurable)

  Size limit: Max 100 entries (oldest removed first)

  Used by:
  - ProductRepository (line 31-36)
  - SalesRepository
  - StoreRepository

  ---
  3. AsyncSQLDatabase

  📁 database/connection/async_sql_database.py:41-352

  Cache implementation:
  self._table_cache: Dict[str, str] = {}

  Apa yang di-cache:
  - Table schema information (CREATE TABLE statements)
  - Column comments, sample rows
  - Cache key: {table_name}:{include_comments}

  Cache methods:
  - Direct dict access (lines 93-97, 140-142)
  - clear_cache() - Manual clear (line 349-352)

  TTL: ❌ TIDAK ADA - Cache persists until manual clear or restart

  Note: Cache ini paling jarang berubah (schema jarang update), jadi no TTL masuk akal

  ---
  📈 Cache Statistics

  | Location           | Type          | TTL   | Size Limit  | Eviction   | Distributed | Persistent |
  |--------------------|---------------|-------|-------------|------------|-------------|------------|
  | AgentConfigManager | Config        | 5 min | ∞           | Time-based | ❌           | ❌          |
  | QueryRepository    | Query Results | 5 min | 100 entries | LRU-like   | ❌           | ❌          |
  | AsyncSQLDatabase   | Table Schema  | ∞     | ∞           | Manual     | ❌           | ❌          |

  ---
  🎯 Recommendation: Redis Migration Strategy

  Priority 1: QueryRepository Cache ⭐⭐⭐

  Why:
  - Most frequent cache hits
  - Directly impacts response time
  - Shared across all agents (product, sales, store)
  - Needs distributed cache untuk horizontal scaling

  Priority 2: AgentConfigManager Cache ⭐⭐

  Why:
  - Config rarely changes but read frequently
  - Benefits from distributed cache
  - Currently has admin endpoints (/cache/invalidate)

  Priority 3: AsyncSQLDatabase Cache ⭐

  Why:
  - Schema changes very rare
  - Low benefit from Redis (already no TTL)
  - Could keep in-memory or move to Redis for consistency

  ---