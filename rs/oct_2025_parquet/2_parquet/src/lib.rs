use parquet_format::{FileMetaData, RowGroup};
use serde_json::{json, Value};

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
