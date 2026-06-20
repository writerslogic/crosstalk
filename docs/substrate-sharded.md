# Design Note: `Sharded<K, V>` Substrate

**Status:** Stub / API surface only (day-2 risk mitigation)
**Crate:** `crosstalk-concurrency`
**Module:** `crosstalk_concurrency::sharded`

## Purpose

This note records the day-2 investigation confirming that a single
`Sharded<K, V>` concurrency primitive can satisfy **two** independent call
sites without divergence:

1. Memory recall (hot-tier)
2. MCP gateway state

No feature implementation is performed here — only the trait/struct
signatures are stubbed so the contract can be reviewed before work begins.

## Common API Surface

```rust
pub struct Sharded<K, V> { /* shard array (impl TBD) */ }

impl<K: Hash + Eq + Clone, V> Sharded<K, V> {
    pub fn new() -> Self;
    pub fn get_or_insert_with<F: FnOnce() -> V>(&self, key: K, init: F) -> Arc<V>;
    pub fn recall(&self, key: &K) -> Option<Arc<V>>;
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>>;
    pub fn get(&self, key: &K) -> Option<Arc<V>>;
}