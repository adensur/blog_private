use parquet_format::FileMetaData;
use serde_json;

use crate::{
    decoders::PageValues,
    errors::Result,
    metadata::{decode_arrow_schema, file_metadata_to_json, row_group_to_json},
    pages::{self, PageChunk, PageContent},
    reader::ParquetFile,
};

pub struct DemoPrinter {
    max_preview_values: usize,
}

impl Default for DemoPrinter {
    fn default() -> Self {
        Self {
            max_preview_values: 5,
        }
    }
}

impl DemoPrinter {
    pub fn new(max_preview_values: usize) -> Self {
        Self { max_preview_values }
    }

    pub fn print_metadata(&self, metadata: &FileMetaData) -> Result<()> {
        println!("\n{}", "=".repeat(80));
        println!("FILE METADATA");
        println!("{}", "=".repeat(80));
        let json_metadata = file_metadata_to_json(metadata);
        println!("{}", serde_json::to_string_pretty(&json_metadata)?);
        Ok(())
    }

    pub fn print_arrow_schema(&self, metadata: &FileMetaData) -> Result<()> {
        if let Some(kv_metadata) = metadata.key_value_metadata.as_ref() {
            for kv in kv_metadata {
                if kv.key == "ARROW:schema" {
                    if let Some(value) = kv.value.as_deref() {
                        println!("\n{}", "=".repeat(80));
                        println!("DECODED ARROW SCHEMA");
                        println!("{}", "=".repeat(80));
                        match decode_arrow_schema(value) {
                            Ok(schema) => {
                                println!("{}", serde_json::to_string_pretty(&schema)?);
                            }
                            Err(e) => {
                                println!("Failed to decode Arrow schema: {}", e);
                            }
                        }
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn print_row_groups(&self, metadata: &FileMetaData) -> Result<()> {
        println!("\n{}", "=".repeat(80));
        println!("ROW GROUPS (Total: {})", metadata.row_groups.len());
        println!("{}", "=".repeat(80));

        for (idx, row_group) in metadata.row_groups.iter().enumerate() {
            println!("\n--- Row Group {} ---", idx);
            let json_rg = row_group_to_json(row_group, idx);
            println!("{}", serde_json::to_string_pretty(&json_rg)?);
        }

        Ok(())
    }

    pub fn print_data_pages(&self, parquet: &mut ParquetFile) -> Result<()> {
        println!("\n{}", "=".repeat(80));
        println!("DATA PAGES");
        println!("{}", "=".repeat(80));

        let metadata = parquet.metadata_arc();

        for (rg_idx, row_group) in metadata.row_groups.iter().enumerate() {
            println!("\n--- Row Group {} Data Pages ---", rg_idx);

            for (col_idx, column_chunk) in row_group.columns.iter().enumerate() {
                let path_str = column_chunk
                    .meta_data
                    .as_ref()
                    .map(|meta| meta.path_in_schema.join("."))
                    .unwrap_or_else(|| "<unknown>".into());
                println!("\n  Column: {} ({})", path_str, col_idx);

                let column_data = pages::read_column_pages(
                    parquet.file_mut(),
                    metadata.as_ref(),
                    column_chunk,
                    col_idx,
                )?;

                for page in &column_data.pages {
                    self.print_page(page, column_data.dictionary.as_ref())?;
                }

                println!("  Total pages read: {}", column_data.page_count);
                println!("  Total values: {}", column_data.total_values);
            }
        }

        Ok(())
    }

    fn print_page(&self, page: &PageChunk, dictionary: Option<&Vec<i64>>) -> Result<()> {
        println!("\n    Page {}:", page.page_number);
        let json_str = serde_json::to_string_pretty(&page.header_json)?;
        for line in json_str.lines() {
            println!("    {}", line);
        }

        match &page.content {
            PageContent::Dictionary { values } => {
                println!("    Dictionary Page Values:");
                self.print_i64_preview(values);
            }
            PageContent::Data { encoding, values } => {
                println!("    Data Page Values (encoding: {:?})", encoding);
                match values {
                    PageValues::Int64(vals) | PageValues::Dictionary(vals) => {
                        self.print_i64_preview(vals);
                    }
                    PageValues::DictionaryIndices(indices) => {
                        if let Some(dict) = dictionary {
                            let mapped: Vec<i64> = indices
                                .iter()
                                .map(|idx| dict.get(*idx as usize).cloned().unwrap_or(0))
                                .collect();
                            println!("      (Dictionary mapped)");
                            self.print_i64_preview(&mapped);
                        } else {
                            println!("      (No dictionary found; printing indices)");
                            self.print_u32_preview(indices);
                        }
                    }
                }
            }
            PageContent::DataV2 => {
                println!("    Data Page V2 (decoding not implemented yet)");
            }
            PageContent::DecodeError(message) => {
                println!("    Failed to decode page: {}", message);
            }
        }

        Ok(())
    }

    fn print_i64_preview(&self, values: &[i64]) {
        for (idx, value) in values.iter().take(self.max_preview_values).enumerate() {
            println!("      [{}]: {}", idx, value);
        }
        if values.len() > self.max_preview_values {
            println!(
                "      ... and {} more",
                values.len() - self.max_preview_values
            );
        }
    }

    fn print_u32_preview(&self, values: &[u32]) {
        for (idx, value) in values.iter().take(self.max_preview_values).enumerate() {
            println!("      [{}]: {}", idx, value);
        }
        if values.len() > self.max_preview_values {
            println!(
                "      ... and {} more",
                values.len() - self.max_preview_values
            );
        }
    }
}
