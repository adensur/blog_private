use parquet_format::{
    ColumnChunk, ColumnMetaData, Encoding, FieldRepetitionType, FileMetaData, PageHeader,
};
use serde_json::Value;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};
use thrift::protocol::TCompactInputProtocol;

use crate::{
    decoders::{self, PageValues},
    errors::{ParquetDemoError, Result},
    metadata,
};

pub struct ColumnData {
    pub pages: Vec<PageChunk>,
    pub total_values: i64,
    pub page_count: usize,
    pub dictionary: Option<Vec<i64>>,
}

pub struct PageChunk {
    pub page_number: usize,
    pub header_json: Value,
    pub content: PageContent,
}

pub enum PageContent {
    Dictionary {
        values: Vec<i64>,
    },
    Data {
        encoding: Encoding,
        values: PageValues,
    },
    DataV2,
    DecodeError(String),
}

pub fn read_column_pages(
    file: &mut File,
    file_metadata: &FileMetaData,
    column_chunk: &ColumnChunk,
    column_idx: usize,
) -> Result<ColumnData> {
    let column_meta = column_chunk
        .meta_data
        .clone()
        .ok_or(ParquetDemoError::MissingColumnMeta { column_idx })?;

    let start_offset = column_meta
        .dictionary_page_offset
        .unwrap_or(column_meta.data_page_offset);
    file.seek(SeekFrom::Start(start_offset as u64))?;

    let mut values_read = 0i64;
    let total_values = column_meta.num_values;
    let def_bit_width = infer_definition_level_bit_width(&column_meta, file_metadata);

    let mut dictionary: Option<Vec<i64>> = None;
    let mut pages = Vec::new();
    let mut page_number = 0usize;

    while values_read < total_values {
        let mut protocol = TCompactInputProtocol::new(&mut *file);
        let page_header = PageHeader::read_from_in_protocol(&mut protocol)?;
        let header_json = metadata::page_header_to_json(&page_header, page_number);

        let mut compressed_data = vec![0u8; page_header.compressed_page_size as usize];
        file.read_exact(&mut compressed_data)?;
        let decompressed = decoders::decompress_page(
            &compressed_data,
            column_meta.codec,
            page_header.uncompressed_page_size,
        )?;

        let content = if let Some(dict_header) = &page_header.dictionary_page_header {
            match decoders::decode_page_data(
                &decompressed,
                column_meta.type_,
                dict_header.encoding,
                dict_header.num_values,
            ) {
                Ok(PageValues::Int64(values)) => {
                    dictionary = Some(values.clone());
                    PageContent::Dictionary { values }
                }
                Ok(other) => {
                    PageContent::DecodeError(format!("Unexpected dictionary payload: {:?}", other))
                }
                Err(e) => PageContent::DecodeError(e.to_string()),
            }
        } else if let Some(data_header) = &page_header.data_page_header {
            let num_values = data_header.num_values as usize;
            let values_slice = if def_bit_width > 0 {
                let (_, consumed) = decoders::decode_rle_bit_packed_hybrid_with_consumed(
                    &decompressed,
                    def_bit_width,
                    num_values,
                )?;
                &decompressed[consumed..]
            } else {
                &decompressed[..]
            };

            match decoders::decode_page_data(
                values_slice,
                column_meta.type_,
                data_header.encoding,
                data_header.num_values,
            ) {
                Ok(page_values) => {
                    values_read += match &page_values {
                        PageValues::Int64(values) => values.len() as i64,
                        PageValues::Dictionary(values) => values.len() as i64,
                        PageValues::DictionaryIndices(values) => values.len() as i64,
                    };
                    PageContent::Data {
                        encoding: data_header.encoding,
                        values: page_values,
                    }
                }
                Err(e) => {
                    values_read += data_header.num_values as i64;
                    PageContent::DecodeError(e.to_string())
                }
            }
        } else if page_header.data_page_header_v2.is_some() {
            values_read += page_header
                .data_page_header_v2
                .as_ref()
                .map(|h| h.num_values as i64)
                .unwrap_or(0);
            PageContent::DataV2
        } else {
            PageContent::DecodeError("Unhandled page type".into())
        };

        pages.push(PageChunk {
            page_number,
            header_json,
            content,
        });
        page_number += 1;
    }

    Ok(ColumnData {
        pages,
        total_values: values_read,
        page_count: page_number,
        dictionary,
    })
}

fn infer_definition_level_bit_width(column_meta: &ColumnMetaData, metadata: &FileMetaData) -> u8 {
    column_meta
        .path_in_schema
        .last()
        .and_then(|name| {
            metadata
                .schema
                .iter()
                .find(|se| se.name == *name)
                .and_then(|se| se.repetition_type)
        })
        .map(|rep| match rep {
            FieldRepetitionType::Optional => 1,
            FieldRepetitionType::Repeated => 1,
            FieldRepetitionType::Required => 0,
        })
        .unwrap_or(0)
}
