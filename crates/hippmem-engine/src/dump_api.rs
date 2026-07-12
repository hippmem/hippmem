//! Engine::dump — full JSONL export API.

use crate::{DumpInput, DumpOutput, EngineResult};
use std::io::Write;

impl crate::Engine {
    /// Exports all MemoryUnit entries as JSON Lines format.
    ///
    /// If `output_path` is Some, writes to a file and returns the path echo;
    /// if None, returns the JSONL string.
    pub fn dump(&self, input: DumpInput) -> EngineResult<DumpOutput> {
        let units = crate::retrieve_api::load_all_units(self.store.db_arc());

        // Build JSONL string
        let mut buf = String::new();
        for unit in &units {
            let line = serde_json::to_string(unit).map_err(|e| {
                crate::EngineError::Internal(format!("failed to serialize MemoryUnit: {}", e))
            })?;
            buf.push_str(&line);
            buf.push('\n');
        }

        let count = units.len() as u64;

        if let Some(path) = input.output_path {
            // Write to file
            let mut file = std::fs::File::create(&path).map_err(|e| {
                crate::EngineError::Store(format!(
                    "cannot create export file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            file.write_all(buf.as_bytes()).map_err(|e| {
                crate::EngineError::Store(format!("failed to write export file: {}", e))
            })?;

            Ok(DumpOutput {
                count,
                written_to: Some(path),
                json: None,
            })
        } else {
            Ok(DumpOutput {
                count,
                written_to: None,
                json: Some(buf),
            })
        }
    }
}
