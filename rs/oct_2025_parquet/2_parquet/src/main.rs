use byteorder::{LittleEndian, ReadBytesExt};
use parquet_format::{FileMetaData, PageHeader};
use parquet_writer::{
    PageValues, decode_arrow_schema, decode_page_data, decode_rle_bit_packed_hybrid_with_consumed,
    decompress_page, file_metadata_to_json, page_header_to_json, row_group_to_json,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use thrift::protocol::TCompactInputProtocol;

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data.parquet";
    let mut file = File::open(file_path)?;

    println!("Reading Parquet file: {}", file_path);

    // Step 1: Read and verify magic header
    let mut header_magic = [0u8; 4];
    file.read_exact(&mut header_magic)?;

    if &header_magic != PARQUET_MAGIC {
        return Err(format!(
            "Invalid Parquet file: expected magic header {:?}, got {:?}",
            PARQUET_MAGIC, header_magic
        )
        .into());
    }
    println!(
        "✓ Valid magic header: {}",
        String::from_utf8_lossy(&header_magic)
    );

    // Step 2: Read and verify magic footer
    file.seek(SeekFrom::End(-4))?;
    let mut footer_magic = [0u8; 4];
    file.read_exact(&mut footer_magic)?;

    if &footer_magic != PARQUET_MAGIC {
        return Err(format!(
            "Invalid Parquet file: expected magic footer {:?}, got {:?}",
            PARQUET_MAGIC, footer_magic
        )
        .into());
    }
    println!(
        "✓ Valid magic footer: {}",
        String::from_utf8_lossy(&footer_magic)
    );

    // Step 3: Read footer length (4 bytes before the magic footer)
    file.seek(SeekFrom::End(-8))?;
    let footer_length = file.read_i32::<LittleEndian>()?;
    println!("✓ Footer length: {} bytes", footer_length);

    if footer_length <= 0 {
        return Err(format!("Invalid footer length: {}", footer_length).into());
    }

    // Step 4: Read FileMetaData
    file.seek(SeekFrom::End(-8 - footer_length as i64))?;
    let mut metadata_bytes = vec![0u8; footer_length as usize];
    file.read_exact(&mut metadata_bytes)?;

    // Step 5: Deserialize FileMetaData using Thrift Compact Protocol
    let mut cursor = std::io::Cursor::new(metadata_bytes);
    let mut protocol: TCompactInputProtocol<&mut std::io::Cursor<Vec<u8>>> =
        TCompactInputProtocol::new(&mut cursor);
    let file_metadata = FileMetaData::read_from_in_protocol(&mut protocol)?;

    // Step 6: Print FileMetaData
    println!("\n{}", "=".repeat(80));
    println!("FILE METADATA");
    println!("{}", "=".repeat(80));
    let json_metadata = file_metadata_to_json(&file_metadata);
    println!("{}", serde_json::to_string_pretty(&json_metadata)?);

    // Step 6.5: Decode and print Arrow schema if present
    if let Some(ref kv_metadata) = file_metadata.key_value_metadata {
        for kv in kv_metadata {
            if kv.key == "ARROW:schema" {
                if let Some(ref schema_b64) = kv.value {
                    println!("\n{}", "=".repeat(80));
                    println!("DECODED ARROW SCHEMA");
                    println!("{}", "=".repeat(80));
                    match decode_arrow_schema(schema_b64) {
                        Ok(arrow_schema) => {
                            println!("{}", serde_json::to_string_pretty(&arrow_schema)?);
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

    // Step 7: Print each RowGroup
    println!("\n{}", "=".repeat(80));
    println!("ROW GROUPS (Total: {})", file_metadata.row_groups.len());
    println!("{}", "=".repeat(80));

    for (i, row_group) in file_metadata.row_groups.iter().enumerate() {
        println!("\n--- Row Group {} ---", i);
        let json_rg = row_group_to_json(row_group, i);
        println!("{}", serde_json::to_string_pretty(&json_rg)?);
    }

    // Step 8: Read and print data pages for each row group
    println!("\n{}", "=".repeat(80));
    println!("DATA PAGES");
    println!("{}", "=".repeat(80));

    for (rg_idx, row_group) in file_metadata.row_groups.iter().enumerate() {
        println!("\n--- Row Group {} Data Pages ---", rg_idx);

        for (col_idx, column_chunk) in row_group.columns.iter().enumerate() {
            let col_meta = column_chunk.meta_data.as_ref().unwrap();
            let path_str = col_meta.path_in_schema.join(".");

            println!("\n  Column: {} ({})", path_str, col_idx);

            // Determine starting offset (dictionary page or data page)
            let start_offset = if let Some(dict_offset) = col_meta.dictionary_page_offset {
                dict_offset
            } else {
                col_meta.data_page_offset
            };

            // Seek to the first page
            file.seek(SeekFrom::Start(start_offset as u64))?;

            let mut values_read = 0;
            let total_values = col_meta.num_values;
            let mut page_num = 0;
            let mut dictionary: Option<Vec<i64>> = None; // Store dictionary for this column

            // Read pages until we've consumed all values
            while values_read < total_values {
                // Read PageHeader
                let mut protocol: TCompactInputProtocol<&mut File> =
                    TCompactInputProtocol::new(&mut file);
                let page_header = PageHeader::read_from_in_protocol(&mut protocol)?;

                // Print page metadata
                println!("\n    Page {}:", page_num);
                let json_page = page_header_to_json(&page_header, page_num);
                let json_str = serde_json::to_string_pretty(&json_page)?;
                // Indent the JSON output
                for line in json_str.lines() {
                    println!("    {}", line);
                }

                // Read compressed page data
                let mut compressed_data = vec![0u8; page_header.compressed_page_size as usize];
                file.read_exact(&mut compressed_data)?;

                // Decompress the page data
                let decompressed = decompress_page(
                    &compressed_data,
                    col_meta.codec,
                    page_header.uncompressed_page_size,
                )?;

                // Decode and print page values
                if let Some(ref dict_header) = page_header.dictionary_page_header {
                    // Dictionary page - store dictionary values
                    println!("    Dictionary Page Values:");
                    match decode_page_data(
                        &decompressed,
                        col_meta.type_,
                        dict_header.encoding,
                        dict_header.num_values,
                    ) {
                        Ok(PageValues::Int64(values)) => {
                            dictionary = Some(values.clone());
                            // Print up to 5 values
                            for (i, val) in values.iter().take(5).enumerate() {
                                println!("      [{}]: {}", i, val);
                            }
                            if values.len() > 5 {
                                println!("      ... and {} more", values.len() - 5);
                            }
                        }
                        Err(e) => {
                            println!("      Failed to decode dictionary: {}", e);
                        }
                        _ => {
                            println!("      Unexpected page value type");
                        }
                    }
                } else if let Some(ref data_header) = page_header.data_page_header {
                    // Data Page V1
                    let num_values = data_header.num_values as usize;
                    println!(
                        "    Data Page Values (encoding: {:?})",
                        data_header.encoding
                    );

                    // Skip repetition levels (bit width 0 in this flat example)
                    // Decode and skip definition levels if present (optional columns have max DL = 1)
                    let def_bit_width: u8 = {
                        // Heuristic: optional leaf -> 1, required -> 0
                        // Look up schema element by last path component
                        let col_name = col_meta.path_in_schema.last().cloned().unwrap_or_default();
                        let rep_type = file_metadata
                            .schema
                            .iter()
                            .find(|se| se.name == col_name)
                            .and_then(|se| se.repetition_type);
                        match rep_type {
                            Some(parquet_format::FieldRepetitionType::Optional) => 1,
                            _ => 0,
                        }
                    };
                    let values_slice = if def_bit_width > 0 {
                        let (_levels, consumed) = match decode_rle_bit_packed_hybrid_with_consumed(
                            &decompressed,
                            def_bit_width,
                            num_values,
                        ) {
                            Ok(x) => x,
                            Err(e) => {
                                println!("      Failed to decode definition levels: {}", e);
                                (Vec::new(), 0usize)
                            }
                        };
                        &decompressed[consumed..]
                    } else {
                        &decompressed[..]
                    };

                    match decode_page_data(
                        values_slice,
                        col_meta.type_,
                        data_header.encoding,
                        data_header.num_values,
                    ) {
                        Ok(PageValues::Int64(values)) => {
                            values_read += values.len() as i64;
                            for (i, val) in values.iter().take(5).enumerate() {
                                println!("      [{}]: {}", i, val);
                            }
                            if values.len() > 5 {
                                println!("      ... and {} more", values.len() - 5);
                            }
                        }
                        Ok(PageValues::DictionaryIndices(indices)) => {
                            values_read += indices.len() as i64;
                            if let Some(ref dict) = dictionary {
                                // Map indices to dictionary values
                                let mapped: Vec<i64> = indices
                                    .iter()
                                    .map(|&idx| {
                                        let i = idx as usize;
                                        if i < dict.len() {
                                            dict[i]
                                        } else {
                                            // Out-of-bounds guard; in well-formed files this shouldn't happen
                                            0
                                        }
                                    })
                                    .collect();
                                for (i, val) in mapped.iter().take(5).enumerate() {
                                    println!("      [{}]: {}", i, val);
                                }
                                if mapped.len() > 5 {
                                    println!("      ... and {} more", mapped.len() - 5);
                                }
                            } else {
                                // No dictionary available, print indices
                                println!("      (No dictionary found; printing indices)");
                                for (i, idx) in indices.iter().take(5).enumerate() {
                                    println!("      [{}]: {}", i, idx);
                                }
                                if indices.len() > 5 {
                                    println!("      ... and {} more", indices.len() - 5);
                                }
                            }
                        }
                        Ok(other) => {
                            values_read += num_values as i64;
                            println!(
                                "      Decoded page values (unsupported display): {:?}",
                                other
                            );
                        }
                        Err(e) => {
                            values_read += num_values as i64;
                            println!("      Failed to decode data page: {}", e);
                        }
                    }
                } else if let Some(ref data_header_v2) = page_header.data_page_header_v2 {
                    values_read += data_header_v2.num_values as i64;
                    println!("    Data Page V2 (decoding not implemented yet)");
                }

                page_num += 1;
            }

            println!("  Total pages read: {}", page_num);
            println!("  Total values: {}", values_read);
        }
    }

    Ok(())
}
