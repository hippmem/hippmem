//! Full-text index: Tantivy BM25 recall channel.
//!
//! Corresponds to ADR-002, the `fulltext/` directory in 04 §5.
//! Chinese tokenization is preprocessed by `hippmem_core::hash::tokenize` (ADR-018); Tantivy handles BM25 scoring.

use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, SchemaBuilder, Value, STORED};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyError};

/// Full-text index: wraps a Tantivy index, providing BM25 recall.
///
/// The index directory is stored under the `fulltext/` subdirectory of the store directory (04 §5).
pub struct FulltextIndex {
    index: Index,
    schema: Schema,
    reader: IndexReader,
    writer: IndexWriter,
    /// Uncommitted document count (batch commit optimization)
    uncommitted: usize,
    /// Auto-commit every N documents (default 1 = commit each one, 10 = batch mode)
    commit_every: usize,
}

impl FulltextIndex {
    /// Creates a new full-text index (creates the directory if it does not exist).
    pub fn create(path: impl AsRef<Path>) -> Result<Self, TantivyError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(|e| {
            TantivyError::SystemError(format!("Failed to create index directory: {e}"))
        })?;
        let schema = build_schema();
        let index = Index::create_in_dir(path, schema.clone())?;
        Self::from_index(index, schema)
    }

    /// Opens an existing full-text index.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TantivyError> {
        let index = Index::open_in_dir(path.as_ref())?;
        let schema = index.schema();
        Self::from_index(index, schema)
    }

    fn from_index(index: Index, schema: Schema) -> Result<Self, TantivyError> {
        let writer = index.writer(50_000_000)?; // 50MB buffer
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            schema,
            reader,
            writer,
            uncommitted: 0,
            commit_every: 1, // Default: commit each one, for backward compatibility
        })
    }

    /// Sets the batch commit interval: auto-commits every `n` documents.
    /// Set to 0 for fully manual commits (call `flush`).
    pub fn set_commit_every(&mut self, n: usize) {
        self.commit_every = n;
    }

    /// Forces committing all unwritten documents.
    pub fn flush(&mut self) -> Result<(), TantivyError> {
        if self.uncommitted > 0 {
            self.writer.commit()?;
            self.uncommitted = 0;
            self.reader.reload()?;
        }
        Ok(())
    }

    /// Adds a document (auto-tokenized).
    ///
    /// - `id`: u128 representation of the MemoryId.
    /// - `text`: raw text (tokenized internally).
    /// - `language`: "zh" or "en", determines the tokenization strategy.
    ///
    /// Whether to commit is decided automatically based on `commit_every` (default: commit each one).
    pub fn add_document(
        &mut self,
        id: u128,
        text: &str,
        language: &str,
    ) -> Result<(), TantivyError> {
        let tokens = hippmem_core::hash::tokenize(text, language);
        self.add_document_tokenized(id, &tokens)
    }

    /// Adds a document (using pre-tokenized tokens, skips the jieba call).
    ///
    /// For batch-write scenarios: the upper layer has already tokenized, so passing tokens directly avoids duplicate jieba overhead.
    pub fn add_document_tokenized(
        &mut self,
        id: u128,
        tokens: &[String],
    ) -> Result<(), TantivyError> {
        let tokenized = tokens.join(" ");

        let body_field = self.schema.get_field("body").unwrap();
        let lo_field = self.schema.get_field("doc_id_lo").unwrap();
        let hi_field = self.schema.get_field("doc_id_hi").unwrap();

        let id_lo = id as u64;
        let id_hi = (id >> 64) as u64;

        self.writer.add_document(doc!(
            lo_field => id_lo,
            hi_field => id_hi,
            body_field => tokenized,
        ))?;

        self.uncommitted += 1;
        if self.commit_every > 0 && self.uncommitted >= self.commit_every {
            self.commit()?;
        }
        Ok(())
    }

    /// Commits unwritten documents so they become searchable. (Kept for backward compatibility)
    pub fn commit(&mut self) -> Result<(), TantivyError> {
        self.writer.commit()?;
        self.uncommitted = 0;
        // Explicitly reload the reader to ensure immediate searchability under the OnCommitWithDelay policy
        self.reader.reload()?;
        Ok(())
    }

    /// Full-text search, returns a list of `(MemoryId/u128, BM25 score)` sorted by score descending.
    pub fn search(&self, query_text: &str, top_k: usize) -> Result<Vec<(u128, f32)>, TantivyError> {
        // Tokenize the query in both Chinese and English and merge, to cover matches in both languages
        let tokens_zh = hippmem_core::hash::tokenize(query_text, "zh");
        let tokens_en = hippmem_core::hash::tokenize(query_text, "en");
        let mut all_tokens: Vec<String> = tokens_zh;
        all_tokens.extend(tokens_en);
        all_tokens.sort();
        all_tokens.dedup();
        let tokenized = all_tokens.join(" ");

        let searcher = self.reader.searcher();
        let body_field = self.schema.get_field("body").unwrap();
        let lo_field = self.schema.get_field("doc_id_lo").unwrap();
        let hi_field = self.schema.get_field("doc_id_hi").unwrap();

        let query_parser = QueryParser::for_index(&self.index, vec![body_field]);
        let query = query_parser.parse_query(&tokenized)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc = searcher.doc::<tantivy::TantivyDocument>(doc_address)?;
            let id_lo = doc
                .get_first(lo_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u128;
            let id_hi = doc
                .get_first(hi_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u128;
            let doc_id = id_lo | (id_hi << 64);
            results.push((doc_id, score));
        }
        Ok(results)
    }
}

/// Builds the Tantivy schema.
fn build_schema() -> Schema {
    let mut builder = SchemaBuilder::new();
    // doc_id_lo: low 64 bits of u128
    builder.add_u64_field("doc_id_lo", STORED);
    // doc_id_hi: high 64 bits of u128
    builder.add_u64_field("doc_id_hi", STORED);
    // body: pre-tokenized text (space-separated), split by non-alphanumeric characters using the default tokenizer
    let text_opts = tantivy::schema::TextOptions::default()
        .set_indexing_options(
            tantivy::schema::TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    builder.add_text_field("body", text_opts);
    builder.build()
}
