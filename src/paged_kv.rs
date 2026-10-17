use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvPageId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvReservationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvCacheConfig {
    pub block_size: usize,
    pub num_pages: usize,
    pub max_sequence_length: usize,
}

impl PagedKvCacheConfig {
    pub fn validate(self) -> Result<(), PagedKvError> {
        if self.block_size == 0 || self.num_pages == 0 || self.max_sequence_length == 0 {
            return Err(PagedKvError(
                "paged KV block_size, num_pages, and max_sequence_length must be non-zero".into(),
            ));
        }
        if self.num_pages > u32::MAX as usize {
            return Err(PagedKvError(
                "paged KV physical page count exceeds u32 page-table capacity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvAppendReservation {
    pub id: KvReservationId,
    pub request_id: RequestId,
    pub start_position: usize,
    pub token_count: usize,
    pub context_length: usize,
    pub newly_allocated_pages: Vec<KvPageId>,
    pub block_table: Vec<KvPageId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvSequenceSnapshot {
    pub request_id: RequestId,
    pub committed_tokens: usize,
    pub block_table: Vec<KvPageId>,
    pub pending_reservation: Option<KvReservationId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvCacheSnapshot {
    pub block_size: usize,
    pub total_pages: usize,
    /// Pages available for allocation, including unreferenced prefix-cache pages.
    pub free_pages: usize,
    pub sequences: Vec<PagedKvSequenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvError(pub String);

impl Display for PagedKvError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PagedKvError {}

#[derive(Debug)]
struct SequenceState {
    committed_tokens: usize,
    pages: Vec<KvPageId>,
    pending: Option<KvReservationId>,
    /// Leading full blocks already published to (or taken from) the prefix cache.
    registered_blocks: usize,
    /// Chain hash of the last registered block.
    chain_hash: u64,
}

#[derive(Debug)]
struct ReservationState {
    request_id: RequestId,
    previous_page_count: usize,
    start_position: usize,
    token_count: usize,
}

/// Identity of one cached full block: the chain hash of every preceding block
/// plus this block's exact tokens.
type BlockKey = (u64, Vec<i32>);

#[derive(Debug, Clone)]
struct CachedBlock {
    key: BlockKey,
    /// Present while the page is unreferenced (eviction order).
    lru_tick: Option<u64>,
}

/// Owns physical-page metadata for every live KV sequence. Appends use an
/// explicit reserve/commit protocol so a failed kernel launch can return all
/// newly acquired pages without exposing a partially advanced sequence.
///
/// Pages are reference counted. With prefix caching enabled, every completed
/// full block of committed tokens is published under a chain key; a new
/// sequence whose tokens start with published blocks maps those physical pages
/// instead of recomputing them. Shared pages are never written: appends always
/// start at `committed_tokens`, which is block aligned after a prefix match.
/// Unreferenced cached pages remain reusable until an allocation evicts them
/// (least recently released first).
#[derive(Debug)]
pub struct PagedKvCacheManager {
    config: PagedKvCacheConfig,
    free_pages: BTreeSet<KvPageId>,
    page_refs: Vec<u32>,
    page_blocks: Vec<Option<CachedBlock>>,
    block_index: HashMap<BlockKey, KvPageId>,
    evictable: BTreeMap<u64, KvPageId>,
    next_tick: u64,
    prefix_caching: bool,
    reused_tokens: u64,
    sequences: BTreeMap<RequestId, SequenceState>,
    reservations: BTreeMap<KvReservationId, ReservationState>,
    next_reservation: u64,
}

const CHAIN_ROOT: u64 = 0x5275_6461_4B56_0001;

fn chain(parent: u64, tokens: &[i32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

impl PagedKvCacheManager {
    pub fn new(config: PagedKvCacheConfig) -> Result<Self, PagedKvError> {
        config.validate()?;
        let free_pages = (0..config.num_pages)
            .map(|page| KvPageId(page as u32))
            .collect();
        Ok(Self {
            config,
            free_pages,
            page_refs: vec![0; config.num_pages],
            page_blocks: vec![None; config.num_pages],
            block_index: HashMap::new(),
            evictable: BTreeMap::new(),
            next_tick: 0,
            prefix_caching: false,
            reused_tokens: 0,
            sequences: BTreeMap::new(),
            reservations: BTreeMap::new(),
            next_reservation: 1,
        })
    }

    pub const fn config(&self) -> PagedKvCacheConfig {
        self.config
    }

    /// Constant-time counters; unlike `snapshot`, these never clone page tables.
    pub fn free_page_count(&self) -> usize {
        self.available_pages()
    }

    pub fn allocated_page_count(&self) -> usize {
        self.config.num_pages - self.available_pages()
    }

    pub fn cached_page_count(&self) -> usize {
        self.evictable.len()
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    /// Enable or disable automatic prefix caching. Disabling drops every
    /// unreferenced cached block; pages mapped by live sequences stay valid.
    pub fn set_prefix_caching(&mut self, enabled: bool) {
        self.prefix_caching = enabled;
        if !enabled {
            let evictable = std::mem::take(&mut self.evictable);
            for page in evictable.into_values() {
                self.drop_cached_block(page);
                self.free_pages.insert(page);
            }
        }
    }

    pub const fn prefix_caching(&self) -> bool {
        self.prefix_caching
    }

    /// Total prompt tokens served from cached pages instead of recomputed.
    pub const fn reused_tokens(&self) -> u64 {
        self.reused_tokens
    }

    /// Pages that an allocation can obtain right now.
    pub fn available_pages(&self) -> usize {
        self.free_pages.len() + self.evictable.len()
    }

    pub fn create_sequence(&mut self, request_id: RequestId) -> Result<(), PagedKvError> {
        if self.sequences.contains_key(&request_id) {
            return Err(PagedKvError(format!(
                "paged KV sequence {} already exists",
                request_id.0
            )));
        }
        self.sequences.insert(
            request_id,
            SequenceState {
                committed_tokens: 0,
                pages: Vec::new(),
                pending: None,
                registered_blocks: 0,
                chain_hash: CHAIN_ROOT,
            },
        );
        Ok(())
    }

    /// Create a sequence and map the longest cached prefix of `tokens` made of
    /// full blocks, always leaving at least one token to compute so the caller
    /// still obtains logits for the final position. Returns the number of
    /// tokens already present in KV (a multiple of `block_size`).
    pub fn create_sequence_with_prefix(
        &mut self,
        request_id: RequestId,
        tokens: &[i32],
    ) -> Result<usize, PagedKvError> {
        self.create_sequence(request_id)?;
        if !self.prefix_caching || tokens.len() <= 1 {
            return Ok(0);
        }
        let block_size = self.config.block_size;
        let usable_blocks = (tokens.len() - 1) / block_size;
        let mut parent = CHAIN_ROOT;
        let mut pages = Vec::new();
        for block in 0..usable_blocks {
            let block_tokens = &tokens[block * block_size..(block + 1) * block_size];
            let key = (parent, block_tokens.to_vec());
            let Some(&page) = self.block_index.get(&key) else {
                break;
            };
            self.acquire_cached(page);
            pages.push(page);
            parent = chain(parent, block_tokens);
        }
        let matched = pages.len() * block_size;
        let sequence = self
            .sequences
            .get_mut(&request_id)
            .expect("sequence was created above");
        sequence.registered_blocks = pages.len();
        sequence.chain_hash = parent;
        sequence.committed_tokens = matched;
        sequence.pages = pages;
        self.reused_tokens += matched as u64;
        Ok(matched)
    }

    /// Publish every newly completed full block of `tokens` (the sequence's
    /// committed tokens, in order) to the prefix cache.
    pub fn register_prefix(
        &mut self,
        request_id: RequestId,
        tokens: &[i32],
    ) -> Result<(), PagedKvError> {
        if !self.prefix_caching {
            return Ok(());
        }
        let block_size = self.config.block_size;
        let sequence = self.sequence(request_id)?;
        let committed = sequence.committed_tokens.min(tokens.len());
        let complete = committed / block_size;
        let mut block = sequence.registered_blocks;
        let mut parent = sequence.chain_hash;
        let pages = sequence.pages.clone();
        while block < complete {
            let block_tokens = &tokens[block * block_size..(block + 1) * block_size];
            let key = (parent, block_tokens.to_vec());
            let hash = chain(parent, block_tokens);
            let page = pages[block];
            if !self.block_index.contains_key(&key) && self.page_blocks[page.0 as usize].is_none() {
                self.block_index.insert(key.clone(), page);
                self.page_blocks[page.0 as usize] = Some(CachedBlock {
                    key,
                    lru_tick: None,
                });
            }
            parent = hash;
            block += 1;
        }
        let sequence = self
            .sequences
            .get_mut(&request_id)
            .expect("sequence was checked above");
        sequence.registered_blocks = block;
        sequence.chain_hash = parent;
        Ok(())
    }

    pub fn contains_sequence(&self, request_id: RequestId) -> bool {
        self.sequences.contains_key(&request_id)
    }

    pub fn committed_tokens(&self, request_id: RequestId) -> Result<usize, PagedKvError> {
        Ok(self.sequence(request_id)?.committed_tokens)
    }

    pub fn block_table(&self, request_id: RequestId) -> Result<Vec<KvPageId>, PagedKvError> {
        Ok(self.sequence(request_id)?.pages.clone())
    }

    pub fn can_append(
        &self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<bool, PagedKvError> {
        if token_count == 0 {
            return Ok(false);
        }
        let sequence = self.sequence(request_id)?;
        if sequence.pending.is_some() {
            return Ok(false);
        }
        let Some(end) = sequence.committed_tokens.checked_add(token_count) else {
            return Ok(false);
        };
        if end > self.config.max_sequence_length {
            return Ok(false);
        }
        let required_pages = end.div_ceil(self.config.block_size);
        Ok(required_pages.saturating_sub(sequence.pages.len()) <= self.available_pages())
    }

    pub fn begin_append(
        &mut self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<PagedKvAppendReservation, PagedKvError> {
        if token_count == 0 {
            return Err(PagedKvError(
                "paged KV append reservation must contain at least one token".into(),
            ));
        }
        let sequence = self.sequence(request_id)?;
        if let Some(reservation) = sequence.pending {
            return Err(PagedKvError(format!(
                "paged KV sequence {} already has pending reservation {}",
                request_id.0, reservation.0
            )));
        }
        let start_position = sequence.committed_tokens;
        let context_length = start_position
            .checked_add(token_count)
            .ok_or_else(|| PagedKvError("paged KV sequence length overflow".into()))?;
        if context_length > self.config.max_sequence_length {
            return Err(PagedKvError(format!(
                "paged KV sequence {} would grow to {context_length}, above capacity {}",
                request_id.0, self.config.max_sequence_length
            )));
        }
        let previous_page_count = sequence.pages.len();
        let required_pages = context_length.div_ceil(self.config.block_size);
        let additional_pages = required_pages.saturating_sub(previous_page_count);
        if additional_pages > self.available_pages() {
            return Err(PagedKvError(format!(
                "paged KV append needs {additional_pages} new pages but only {} are free",
                self.available_pages()
            )));
        }
        let id = KvReservationId(self.next_reservation);
        self.next_reservation = self
            .next_reservation
            .checked_add(1)
            .ok_or_else(|| PagedKvError("paged KV reservation id overflow".into()))?;

        let mut newly_allocated_pages = Vec::with_capacity(additional_pages);
        for _ in 0..additional_pages {
            newly_allocated_pages.push(self.allocate_page());
        }
        let sequence = self
            .sequences
            .get_mut(&request_id)
            .expect("sequence was checked before allocation");
        sequence.pages.extend(newly_allocated_pages.iter().copied());
        sequence.pending = Some(id);
        let block_table = sequence.pages.clone();
        self.reservations.insert(
            id,
            ReservationState {
                request_id,
                previous_page_count,
                start_position,
                token_count,
            },
        );
        Ok(PagedKvAppendReservation {
            id,
            request_id,
            start_position,
            token_count,
            context_length,
            newly_allocated_pages,
            block_table,
        })
    }

    pub fn commit_append(
        &mut self,
        reservation_id: KvReservationId,
    ) -> Result<usize, PagedKvError> {
        self.validate_reservation(reservation_id)?;
        let reservation = self.take_reservation(reservation_id)?;
        let sequence = self
            .sequences
            .get_mut(&reservation.request_id)
            .ok_or_else(|| PagedKvError("reservation references a removed sequence".into()))?;
        if sequence.pending != Some(reservation_id) {
            return Err(PagedKvError(format!(
                "reservation {} does not own sequence {} pending state",
                reservation_id.0, reservation.request_id.0
            )));
        }
        let context_length = reservation
            .start_position
            .checked_add(reservation.token_count)
            .ok_or_else(|| PagedKvError("paged KV committed length overflow".into()))?;
        sequence.committed_tokens = context_length;
        sequence.pending = None;
        Ok(context_length)
    }

    /// Extend an in-flight append for speculative device work. The caller must
    /// synchronize that work before committing a prefix or releasing pages.
    pub fn extend_append(&mut self,id:KvReservationId,extra:usize)->Result<PagedKvAppendReservation,PagedKvError> {
        self.validate_reservation(id)?;
        let reservation=&self.reservations[&id];
        let request_id=reservation.request_id;let start=reservation.start_position;
        let count=reservation.token_count.checked_add(extra).ok_or_else(||PagedKvError("lookahead token count overflow".into()))?;
        let end=start.checked_add(count).ok_or_else(||PagedKvError("lookahead context overflow".into()))?;
        if end>self.config.max_sequence_length {return Err(PagedKvError("lookahead exceeds sequence capacity".into()));}
        let current=self.sequences[&request_id].pages.len();
        let additional=end.div_ceil(self.config.block_size).saturating_sub(current);
        if additional>self.available_pages() {return Err(PagedKvError("insufficient lookahead pages".into()));}
        let mut newly_allocated_pages=Vec::with_capacity(additional);
        for _ in 0..additional {newly_allocated_pages.push(self.allocate_page());}
        let sequence=self.sequences.get_mut(&request_id).unwrap();sequence.pages.extend(&newly_allocated_pages);
        self.reservations.get_mut(&id).unwrap().token_count=count;
        Ok(PagedKvAppendReservation{id,request_id,start_position:start,token_count:count,context_length:end,
            newly_allocated_pages,block_table:sequence.pages.clone()})
    }

    /// Commit only the consumed inputs, atomically validating every row first.
    /// Unconsumed lookahead pages return to the same bounded physical pool.
    pub fn commit_append_prefixes(&mut self,rows:&[(KvReservationId,usize)])->Result<Vec<usize>,PagedKvError> {
        self.validate_reservations(&rows.iter().map(|(id,_)|*id).collect::<Vec<_>>())?;
        for &(id,count) in rows {
            if count==0 || count>self.reservations[&id].token_count {return Err(PagedKvError("invalid consumed lookahead prefix".into()));}
        }
        let mut positions=Vec::with_capacity(rows.len());
        for &(id,count) in rows {
            let reservation=self.reservations.remove(&id).unwrap();
            let sequence=self.sequences.get_mut(&reservation.request_id).unwrap();
            let position=reservation.start_position+count;
            let released=sequence.pages.split_off(position.div_ceil(self.config.block_size));
            sequence.committed_tokens=position;sequence.pending=None;positions.push(position);
            for page in released { self.release_page(page); }
        }
        Ok(positions)
    }

    /// Commit only the first `token_count` tokens of a reservation (for
    /// example the accepted prefix of a speculative verification) and release
    /// reserved pages that the shorter sequence no longer needs. Slots past the
    /// committed length may hold stale values; later appends overwrite them.
    pub fn commit_append_partial(
        &mut self,
        reservation_id: KvReservationId,
        token_count: usize,
    ) -> Result<usize, PagedKvError> {
        let reservation = self.reservations.get(&reservation_id).ok_or_else(|| {
            PagedKvError(format!(
                "paged KV reservation {} does not exist",
                reservation_id.0
            ))
        })?;
        if token_count == 0 || token_count > reservation.token_count {
            return Err(PagedKvError(format!(
                "partial commit of {token_count} tokens is outside 1..={}",
                reservation.token_count
            )));
        }
        let request_id = reservation.request_id;
        let context_length = reservation.start_position + token_count;
        let needed_pages = context_length.div_ceil(self.config.block_size);
        let committed = self.commit_append(reservation_id)?;
        let sequence = self
            .sequences
            .get_mut(&request_id)
            .expect("commit validated the sequence");
        sequence.committed_tokens = context_length;
        debug_assert!(committed >= context_length);
        let released = if sequence.pages.len() > needed_pages {
            sequence.pages.split_off(needed_pages)
        } else {
            Vec::new()
        };
        for page in released {
            self.release_page(page);
        }
        Ok(context_length)
    }

    pub fn cancel_append(&mut self, reservation_id: KvReservationId) -> Result<(), PagedKvError> {
        self.validate_reservation(reservation_id)?;
        let reservation = self.take_reservation(reservation_id)?;
        let sequence = self
            .sequences
            .get_mut(&reservation.request_id)
            .ok_or_else(|| PagedKvError("reservation references a removed sequence".into()))?;
        if sequence.pending != Some(reservation_id) {
            return Err(PagedKvError(format!(
                "reservation {} does not own sequence {} pending state",
                reservation_id.0, reservation.request_id.0
            )));
        }
        let released = sequence.pages.split_off(reservation.previous_page_count);
        sequence.pending = None;
        for page in released {
            self.release_page(page);
        }
        Ok(())
    }

    /// Commit a group atomically with respect to validation errors. Unknown or
    /// duplicate IDs leave every reservation, sequence and page unchanged.
    pub fn commit_appends(
        &mut self,
        reservation_ids: &[KvReservationId],
    ) -> Result<Vec<usize>, PagedKvError> {
        self.validate_reservations(reservation_ids)?;
        Ok(reservation_ids.iter().map(|&id| {
            self.commit_append(id).expect("all reservations were validated before commit")
        }).collect())
    }

    /// Cancel a group atomically with respect to validation errors.
    pub fn cancel_appends(
        &mut self,
        reservation_ids: &[KvReservationId],
    ) -> Result<(), PagedKvError> {
        self.validate_reservations(reservation_ids)?;
        for &id in reservation_ids {
            self.cancel_append(id).expect("all reservations were validated before cancellation");
        }
        Ok(())
    }

    fn validate_reservations(&self, ids: &[KvReservationId]) -> Result<(), PagedKvError> {
        let mut seen = BTreeSet::new();
        for &id in ids {
            if !seen.insert(id) {
                return Err(PagedKvError(format!("duplicate reservation {}", id.0)));
            }
            self.validate_reservation(id)?;
        }
        Ok(())
    }

    fn validate_reservation(&self, id: KvReservationId) -> Result<(), PagedKvError> {
        let reservation = self.reservations.get(&id).ok_or_else(|| {
            PagedKvError(format!("paged KV reservation {} does not exist", id.0))
        })?;
        let sequence = self.sequence(reservation.request_id)?;
        if sequence.pending != Some(id)
            || sequence.committed_tokens != reservation.start_position
            || sequence.pages.len() < reservation.previous_page_count
        {
            return Err(PagedKvError(format!(
                "reservation {} does not own a consistent sequence state", id.0
            )));
        }
        reservation.start_position.checked_add(reservation.token_count)
            .ok_or_else(|| PagedKvError("paged KV committed length overflow".into()))?;
        Ok(())
    }

    pub fn remove_sequence(&mut self, request_id: RequestId) -> Result<(), PagedKvError> {
        let sequence = self.sequence(request_id)?;
        if let Some(reservation) = sequence.pending {
            return Err(PagedKvError(format!(
                "cannot remove sequence {} while reservation {} is pending",
                request_id.0, reservation.0
            )));
        }
        let sequence = self
            .sequences
            .remove(&request_id)
            .expect("sequence was checked before removal");
        // Release the tail first so earlier (more widely shared) blocks are
        // the last to become eviction candidates.
        for page in sequence.pages.into_iter().rev() {
            self.release_page(page);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> PagedKvCacheSnapshot {
        PagedKvCacheSnapshot {
            block_size: self.config.block_size,
            total_pages: self.config.num_pages,
            free_pages: self.available_pages(),
            sequences: self
                .sequences
                .iter()
                .map(|(&request_id, sequence)| PagedKvSequenceSnapshot {
                    request_id,
                    committed_tokens: sequence.committed_tokens,
                    block_table: sequence.pages.clone(),
                    pending_reservation: sequence.pending,
                })
                .collect(),
        }
    }

    fn allocate_page(&mut self) -> KvPageId {
        let page = match self.free_pages.pop_first() {
            Some(page) => page,
            None => {
                let (_, page) = self
                    .evictable
                    .pop_first()
                    .expect("available-page count was checked before allocation");
                self.drop_cached_block(page);
                page
            }
        };
        debug_assert_eq!(self.page_refs[page.0 as usize], 0);
        self.page_refs[page.0 as usize] = 1;
        page
    }

    fn acquire_cached(&mut self, page: KvPageId) {
        let index = page.0 as usize;
        if self.page_refs[index] == 0 {
            let block = self.page_blocks[index]
                .as_mut()
                .expect("indexed page carries its cached block");
            if let Some(tick) = block.lru_tick.take() {
                self.evictable.remove(&tick);
            }
        }
        self.page_refs[index] += 1;
    }

    fn release_page(&mut self, page: KvPageId) {
        let index = page.0 as usize;
        debug_assert!(self.page_refs[index] > 0, "released page must be referenced");
        self.page_refs[index] -= 1;
        if self.page_refs[index] != 0 {
            return;
        }
        if self.prefix_caching && self.page_blocks[index].is_some() {
            let tick = self.next_tick;
            self.next_tick += 1;
            self.page_blocks[index]
                .as_mut()
                .expect("checked above")
                .lru_tick = Some(tick);
            self.evictable.insert(tick, page);
        } else {
            self.drop_cached_block(page);
            let inserted = self.free_pages.insert(page);
            debug_assert!(inserted, "released page must not already be free");
        }
    }

    fn drop_cached_block(&mut self, page: KvPageId) {
        if let Some(block) = self.page_blocks[page.0 as usize].take() {
            if self.block_index.get(&block.key) == Some(&page) {
                self.block_index.remove(&block.key);
            }
        }
    }

    fn sequence(&self, request_id: RequestId) -> Result<&SequenceState, PagedKvError> {
        self.sequences.get(&request_id).ok_or_else(|| {
            PagedKvError(format!("paged KV sequence {} does not exist", request_id.0))
        })
    }

    fn take_reservation(
        &mut self,
        reservation_id: KvReservationId,
    ) -> Result<ReservationState, PagedKvError> {
        self.reservations.remove(&reservation_id).ok_or_else(|| {
            PagedKvError(format!(
                "paged KV reservation {} does not exist",
                reservation_id.0
            ))
        })
    }
}

#[cfg(test)]
mod tests;

impl PagedKvCacheConfig {
    /// Size a dense cache AFTER accounting for loaded weights. Does not query
    /// VRAM or allocate device buffers; physical allocation can still fail.
    pub fn from_memory_budget(
        geometry: crate::runtime::KvMemoryGeometry,
        available_bytes: u64,
        workspace_bytes: u64,
        headroom_bytes: u64,
        max_sequence_length: usize,
    ) -> Result<Self, PagedKvError> {
        let num_pages = geometry.page_capacity(available_bytes, workspace_bytes, headroom_bytes)
            .map_err(|e| PagedKvError(e.to_string()))?;
        let config = Self { block_size: geometry.tokens_per_page, num_pages, max_sequence_length };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    fn manager(num_pages: usize) -> PagedKvCacheManager {
        let mut manager = PagedKvCacheManager::new(PagedKvCacheConfig {
            block_size: 2,
            num_pages,
            max_sequence_length: 16,
        })
        .unwrap();
        manager.set_prefix_caching(true);
        manager
    }

    fn fill(manager: &mut PagedKvCacheManager, id: RequestId, tokens: &[i32]) {
        let matched = manager.create_sequence_with_prefix(id, tokens).unwrap();
        let reservation = manager.begin_append(id, tokens.len() - matched).unwrap();
        manager.commit_append(reservation.id).unwrap();
        manager.register_prefix(id, tokens).unwrap();
    }

    #[test]
    fn shared_prefix_maps_the_same_pages_and_survives_the_first_owner() {
        let mut manager = manager(8);
        fill(&mut manager, RequestId(1), &[1, 2, 3, 4, 5]);
        let first = manager.block_table(RequestId(1)).unwrap();
        let matched = manager
            .create_sequence_with_prefix(RequestId(2), &[1, 2, 3, 4, 9, 9])
            .unwrap();
        assert_eq!(matched, 4);
        assert_eq!(manager.block_table(RequestId(2)).unwrap(), first[..2]);
        manager.remove_sequence(RequestId(1)).unwrap();
        // Pages still referenced by sequence 2 are neither free nor cached.
        assert_eq!(manager.snapshot().free_pages, 8 - 2);
        manager.remove_sequence(RequestId(2)).unwrap();
        assert_eq!(manager.snapshot().free_pages, 8);
        assert_eq!(manager.cached_page_count(), 2);
        let matched = manager
            .create_sequence_with_prefix(RequestId(3), &[1, 2, 3, 4, 7])
            .unwrap();
        assert_eq!(matched, 4);
        assert_eq!(manager.reused_tokens(), 8);
    }

    #[test]
    fn eviction_reclaims_cached_pages_and_forgets_them() {
        let mut manager = manager(3);
        fill(&mut manager, RequestId(1), &[1, 2, 3, 4, 5]);
        manager.remove_sequence(RequestId(1)).unwrap();
        assert_eq!(manager.cached_page_count(), 2);
        fill(&mut manager, RequestId(2), &[8, 8, 8, 8, 8, 8]);
        assert_eq!(manager.snapshot().free_pages, 0);
        manager.remove_sequence(RequestId(2)).unwrap();
        let matched = manager
            .create_sequence_with_prefix(RequestId(3), &[1, 2, 3, 4, 5])
            .unwrap();
        assert_eq!(matched, 0, "evicted blocks must not be matched");
    }

    #[test]
    fn whole_prompt_match_leaves_one_token_to_compute() {
        let mut manager = manager(8);
        fill(&mut manager, RequestId(1), &[1, 2, 3, 4]);
        let matched = manager
            .create_sequence_with_prefix(RequestId(2), &[1, 2, 3, 4])
            .unwrap();
        assert_eq!(matched, 2);
    }
}
