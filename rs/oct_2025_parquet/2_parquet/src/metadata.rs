use base64::prelude::*;
use parquet_format::{FileMetaData, PageHeader, RowGroup};
use serde_json::{Value, json};

use crate::errors::{ParquetDemoError, Result};

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
pub fn decode_arrow_schema(base64_str: &str) -> Result<Value> {
    let decoded = BASE64_STANDARD
        .decode(base64_str)
        .map_err(|e| ParquetDemoError::ArrowSchema(format!("Base64 decode failed: {}", e)))?;

    let data = if decoded.len() >= 8 {
        let continuation = u32::from_le_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]);
        if continuation == 0xFFFFFFFF {
            &decoded[8..]
        } else {
            &decoded[..]
        }
    } else {
        &decoded[..]
    };

    let message = arrow_ipc::root_as_message(data)
        .map_err(|e| ParquetDemoError::ArrowSchema(format!("Arrow IPC parse failed: {}", e)))?;

    let ipc_schema = message
        .header_as_schema()
        .ok_or_else(|| ParquetDemoError::ArrowSchema("IPC message lacks schema".into()))?;

    let schema = arrow_ipc::convert::fb_to_schema(ipc_schema);

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
