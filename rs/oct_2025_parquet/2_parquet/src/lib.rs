use base64::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt};
use parquet_format::{CompressionCodec, Encoding, FileMetaData, PageHeader, RowGroup, Type};
use serde_json::{Value, json};
use std::io::{Cursor, Read};

/// Convert FileMetaData to human-readable JSON with enum names
pub fn file_metadata_to_json(metadata: &FileMetaData) -> Value {
    json!({
        "version": metadata.version,
        "num_rows": metadata.num_rows,
        "created_by": metadata.created_by,
        "num_row_groups": metadata.row_groups.len(),
        "schema": metadata.schema.iter().map(|se| {
            json!({
                "type": format!("{:?}", se.type_),
                "type_length": se.type_length,
                "repetition_type": format!("{:?}", se.repetition_type),
                "name": se.name,
                "num_children": se.num_children,
                "converted_type": format!("{:?}", se.converted_type),
                "scale": se.scale,
                "precision": se.precision,
                "field_id": se.field_id,
                "logical_type": format!("{:?}", se.logical_type),
            })
        }).collect::<Vec<_>>(),
        "key_value_metadata": metadata.key_value_metadata.as_ref().map(|kv_list| {
            kv_list.iter().map(|kv| {
                json!({
                    "key": kv.key,
                    "value": kv.value,
                })
            }).collect::<Vec<_>>()
        }),
        "column_orders": metadata.column_orders.as_ref().map(|orders| {
            orders.iter().map(|order| format!("{:?}", order)).collect::<Vec<_>>()
        }),
        "encryption_algorithm": format!("{:?}", metadata.encryption_algorithm),
        "footer_signing_key_metadata": metadata.footer_signing_key_metadata.as_ref().map(hex::encode),
    })
}

/// Convert RowGroup to human-readable JSON with enum names
pub fn row_group_to_json(rg: &RowGroup, index: usize) -> Value {
    json!({
        "row_group_index": index,
        "num_rows": rg.num_rows,
        "total_byte_size": rg.total_byte_size,
        "num_columns": rg.columns.len(),
        "columns": rg.columns.iter().map(|col| {
            json!({
                "file_path": col.file_path,
                "file_offset": col.file_offset,
                "metadata": {
                    "type": format!("{:?}", col.meta_data.as_ref().unwrap().type_),
                    "encodings": col.meta_data.as_ref().unwrap().encodings.iter()
                        .map(|e| format!("{:?}", e)).collect::<Vec<_>>(),
                    "path_in_schema": col.meta_data.as_ref().unwrap().path_in_schema,
                    "codec": format!("{:?}", col.meta_data.as_ref().unwrap().codec),
                    "num_values": col.meta_data.as_ref().unwrap().num_values,
                    "total_uncompressed_size": col.meta_data.as_ref().unwrap().total_uncompressed_size,
                    "total_compressed_size": col.meta_data.as_ref().unwrap().total_compressed_size,
                    "data_page_offset": col.meta_data.as_ref().unwrap().data_page_offset,
                    "index_page_offset": col.meta_data.as_ref().unwrap().index_page_offset,
                    "dictionary_page_offset": col.meta_data.as_ref().unwrap().dictionary_page_offset,
                    "statistics": col.meta_data.as_ref().unwrap().statistics.as_ref().map(|stats| {
                        json!({
                            "max": stats.max_value.as_ref().map(hex::encode),
                            "min": stats.min_value.as_ref().map(hex::encode),
                            "null_count": stats.null_count,
                            "distinct_count": stats.distinct_count,
                        })
                    }),
                }
            })
        }).collect::<Vec<_>>(),
    })
}

/// Convert PageHeader to human-readable JSON with enum names
pub fn page_header_to_json(header: &PageHeader, page_num: usize) -> Value {
    let mut json_obj = json!({
        "page_number": page_num,
        "type": format!("{:?}", header.type_),
        "uncompressed_page_size": header.uncompressed_page_size,
        "compressed_page_size": header.compressed_page_size,
        "crc": header.crc,
    });

    // Add type-specific headers
    if let Some(ref data_page_header) = header.data_page_header {
        json_obj["data_page_header"] = json!({
            "num_values": data_page_header.num_values,
            "encoding": format!("{:?}", data_page_header.encoding),
            "definition_level_encoding": format!("{:?}", data_page_header.definition_level_encoding),
            "repetition_level_encoding": format!("{:?}", data_page_header.repetition_level_encoding),
            "statistics": data_page_header.statistics.as_ref().map(|stats| {
                json!({
                    "max": stats.max_value.as_ref().map(hex::encode),
                    "min": stats.min_value.as_ref().map(hex::encode),
                    "null_count": stats.null_count,
                    "distinct_count": stats.distinct_count,
                })
            }),
        });
    }

    if let Some(ref dict_page_header) = header.dictionary_page_header {
        json_obj["dictionary_page_header"] = json!({
            "num_values": dict_page_header.num_values,
            "encoding": format!("{:?}", dict_page_header.encoding),
            "is_sorted": dict_page_header.is_sorted,
        });
    }

    if let Some(ref data_page_header_v2) = header.data_page_header_v2 {
        json_obj["data_page_header_v2"] = json!({
            "num_values": data_page_header_v2.num_values,
            "num_nulls": data_page_header_v2.num_nulls,
            "num_rows": data_page_header_v2.num_rows,
            "encoding": format!("{:?}", data_page_header_v2.encoding),
            "definition_levels_byte_length": data_page_header_v2.definition_levels_byte_length,
            "repetition_levels_byte_length": data_page_header_v2.repetition_levels_byte_length,
            "is_compressed": data_page_header_v2.is_compressed,
            "statistics": data_page_header_v2.statistics.as_ref().map(|stats| {
                json!({
                    "max": stats.max_value.as_ref().map(hex::encode),
                    "min": stats.min_value.as_ref().map(hex::encode),
                    "null_count": stats.null_count,
                    "distinct_count": stats.distinct_count,
                })
            }),
        });
    }

    json_obj
}

/// Decode Arrow IPC schema from base64-encoded string
pub fn decode_arrow_schema(base64_str: &str) -> Result<Value, Box<dyn std::error::Error>> {
    // Decode base64
    let decoded = BASE64_STANDARD.decode(base64_str)?;

    // Arrow IPC format may have:
    // - 4 bytes: continuation marker (0xFFFFFFFF)
    // - 4 bytes: metadata length
    // - N bytes: flatbuffer message

    // Check for continuation marker
    let data = if decoded.len() >= 8 {
        let continuation = u32::from_le_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]);
        if continuation == 0xFFFFFFFF {
            // Has continuation marker, skip it and the length field
            let _length = u32::from_le_bytes([decoded[4], decoded[5], decoded[6], decoded[7]]);
            &decoded[8..]
        } else {
            // No continuation marker, use as-is
            &decoded[..]
        }
    } else {
        &decoded[..]
    };

    // Parse Arrow IPC message using arrow-ipc
    use arrow_ipc::root_as_message;

    let message =
        root_as_message(data).map_err(|e| format!("Failed to parse Arrow IPC message: {}", e))?;

    // Extract schema from the message
    let ipc_schema = message
        .header_as_schema()
        .ok_or("Message does not contain a schema")?;

    // Convert to arrow Schema
    let schema = arrow_ipc::convert::fb_to_schema(ipc_schema);

    // Convert to JSON
    Ok(json!({
        "fields": schema.fields().iter().map(|field| {
            json!({
                "name": field.name(),
                "data_type": format!("{:?}", field.data_type()),
                "nullable": field.is_nullable(),
                "metadata": field.metadata(),
            })
        }).collect::<Vec<_>>(),
        "metadata": schema.metadata(),
    }))
}

/// Decompress page data based on the codec
pub fn decompress_page(
    compressed_data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: i32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(compressed_data.to_vec()),
        CompressionCodec::Snappy => Err("Snappy decompression not implemented".into()),
        CompressionCodec::Gzip => Err("Gzip decompression not implemented".into()),
        CompressionCodec::Lzo => Err("LZO decompression not implemented".into()),
        CompressionCodec::Brotli => Err("Brotli decompression not implemented".into()),
        CompressionCodec::Lz4 => Err("LZ4 decompression not implemented".into()),
        CompressionCodec::Zstd => {
            let decompressed = zstd::decode_all(compressed_data)?;
            if decompressed.len() != uncompressed_size as usize {
                return Err(format!(
                    "Decompressed size mismatch: expected {}, got {}",
                    uncompressed_size,
                    decompressed.len()
                )
                .into());
            }
            Ok(decompressed)
        }
    }
}

/// Decode Plain-encoded Int64 values
pub fn decode_plain_int64(
    data: &[u8],
    count: usize,
) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(data);
    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        let value = cursor.read_i64::<LittleEndian>()?;
        values.push(value);
    }

    Ok(values)
}

/// Decode RLE bit-packed hybrid for definition levels
/// This is a simplified version that handles the common case
pub fn decode_rle_bit_packed_hybrid(
    data: &[u8],
    bit_width: u8,
    max_values: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (values, _consumed) =
        decode_rle_bit_packed_hybrid_with_consumed(data, bit_width, max_values)?;
    Ok(values)
}

/// Decode RLE bit-packed hybrid and also return how many bytes were consumed
/// Simplified: optimized for bit_width <= 1 (definition levels in common cases)
pub fn decode_rle_bit_packed_hybrid_with_consumed(
    data: &[u8],
    bit_width: u8,
    max_values: usize,
) -> Result<(Vec<u8>, usize), Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(data);
    let mut result = Vec::new();

    while result.len() < max_values {
        // Read the run header (varint)
        let header = read_varint(&mut cursor)?;

        if header & 1 == 0 {
            // RLE run
            let run_length = (header >> 1) as usize;
            let value = if bit_width > 0 {
                read_bits(&mut cursor, bit_width)?
            } else {
                0
            };

            for _ in 0..run_length {
                result.push(value);
                if result.len() >= max_values {
                    break;
                }
            }
        } else {
            // Bit-packed run
            let count = (header >> 1) as usize;
            for _ in 0..count * 8 {
                if result.len() >= max_values {
                    break;
                }
                let value = read_bits(&mut cursor, bit_width)?;
                result.push(value);
            }
        }
    }

    Ok((result, cursor.position() as usize))
}

/// Read a varint from the cursor
fn read_varint(cursor: &mut Cursor<&[u8]>) -> Result<u32, Box<dyn std::error::Error>> {
    let mut result = 0u32;
    let mut shift = 0;

    loop {
        let byte = cursor.read_u8()?;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }

    Ok(result)
}

/// Read a specific number of bits (for RLE decoding)
/// This is a simplified implementation that assumes bit_width is 1
fn read_bits(cursor: &mut Cursor<&[u8]>, bit_width: u8) -> Result<u8, Box<dyn std::error::Error>> {
    if bit_width == 0 {
        return Ok(0);
    }

    // For bit_width of 1, just read one bit at a time
    // This is simplified - a real implementation would handle bit packing properly
    if bit_width == 1 {
        let byte = cursor.read_u8()?;
        Ok(byte)
    } else {
        // For other bit widths, read a byte
        let byte = cursor.read_u8()?;
        Ok(byte & ((1 << bit_width) - 1))
    }
}

/// Decode dictionary-encoded values (returns dictionary indices)
/// Simplified version that handles RLE runs
pub fn decode_rle_dictionary(
    data: &[u8],
    count: usize,
    bit_width: u8,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Dictionary encoding: RLE/bit-packed hybrid
    // Format: varint header indicating run length and type

    let mut cursor = Cursor::new(data);
    let mut result = Vec::new();

    while result.len() < count {
        // Read varint header
        let header = match read_varint(&mut cursor) {
            Ok(h) => h,
            Err(_) => break, // End of data
        };

        if header & 1 == 0 {
            // RLE run: repeated value
            let run_length = (header >> 1) as usize;

            // Read the repeated value (bit_width bits)
            // For simplicity, if bit_width <= 8, read as bytes
            let value = if bit_width <= 8 {
                let bytes_needed = (bit_width as usize).div_ceil(8);
                let mut val = 0u32;
                for i in 0..bytes_needed {
                    if let Ok(byte) = cursor.read_u8() {
                        val |= (byte as u32) << (i * 8);
                    }
                }
                val & ((1 << bit_width) - 1)
            } else {
                return Err("Bit width > 8 not supported".into());
            };

            for _ in 0..run_length {
                result.push(value);
                if result.len() >= count {
                    break;
                }
            }
        } else {
            // Bit-packed run
            let count_groups = (header >> 1) as usize;

            // Each group contains 8 values bit-packed
            // We need to read them bit by bit across byte boundaries
            let values_to_read = std::cmp::min(count_groups * 8, count - result.len());
            let bytes_needed = (values_to_read * bit_width as usize).div_ceil(8);

            // Read all bytes for this bit-packed run
            let mut packed_bytes = vec![0u8; bytes_needed];
            cursor.read_exact(&mut packed_bytes)?;

            // Decode bit-packed values
            let mut bit_offset = 0;
            for _ in 0..values_to_read {
                if result.len() >= count {
                    break;
                }

                // Extract bit_width bits starting at bit_offset
                let mut value = 0u32;
                for bit_idx in 0..bit_width {
                    let total_bit_pos = bit_offset + bit_idx as usize;
                    let byte_pos = total_bit_pos / 8;
                    let bit_pos_in_byte = total_bit_pos % 8;

                    if byte_pos < packed_bytes.len() {
                        let bit = (packed_bytes[byte_pos] >> bit_pos_in_byte) & 1;
                        value |= (bit as u32) << bit_idx;
                    }
                }

                result.push(value);
                bit_offset += bit_width as usize;
            }
        }
    }

    Ok(result)
}

/// Enum for decoded page values
#[derive(Debug)]
pub enum PageValues {
    Int64(Vec<i64>),
    Dictionary(Vec<i64>),        // Dictionary values
    DictionaryIndices(Vec<u32>), // Indices into dictionary
}

/// Decode page data based on type and encoding
pub fn decode_page_data(
    page_data: &[u8],
    page_type: Type,
    encoding: Encoding,
    num_values: i32,
) -> Result<PageValues, Box<dyn std::error::Error>> {
    match (page_type, encoding) {
        (Type::Int64, Encoding::Plain) => {
            let values = decode_plain_int64(page_data, num_values as usize)?;
            Ok(PageValues::Int64(values))
        }
        (Type::Int64, Encoding::RleDictionary) => {
            // For dictionary encoding, we need the bit width
            let bit_width = page_data[0];
            let indices = if page_data.len() > 1 {
                decode_rle_dictionary(&page_data[1..], num_values as usize, bit_width)?
            } else {
                Vec::new()
            };
            Ok(PageValues::DictionaryIndices(indices))
        }
        _ => Err(format!(
            "Decoding not implemented for type {:?} with encoding {:?}",
            page_type, encoding
        )
        .into()),
    }
}

/// Decode page data with explicit bit width for dictionary encoding
pub fn decode_page_data_with_dict_info(
    page_data: &[u8],
    page_type: Type,
    encoding: Encoding,
    num_values: i32,
    dict_bit_width: u8,
) -> Result<PageValues, Box<dyn std::error::Error>> {
    match (page_type, encoding) {
        (Type::Int64, Encoding::Plain) => {
            let values = decode_plain_int64(page_data, num_values as usize)?;
            Ok(PageValues::Int64(values))
        }
        (Type::Int64, Encoding::RleDictionary) => {
            // Use provided bit width instead of reading from data
            let indices = decode_rle_dictionary(page_data, num_values as usize, dict_bit_width)?;
            Ok(PageValues::DictionaryIndices(indices))
        }
        _ => Err(format!(
            "Decoding not implemented for type {:?} with encoding {:?}",
            page_type, encoding
        )
        .into()),
    }
}
