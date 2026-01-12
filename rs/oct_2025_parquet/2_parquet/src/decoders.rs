use byteorder::{LittleEndian, ReadBytesExt};
use parquet_format::{CompressionCodec, Encoding, Type};
use std::io::{Cursor, Read};

use crate::errors::{ParquetDemoError, Result};

/// Enum for decoded page values
#[derive(Debug, Clone)]
pub enum PageValues {
    Int64(Vec<i64>),
    Dictionary(Vec<i64>),
    DictionaryIndices(Vec<u32>),
}

/// Decompress page data based on the codec
pub fn decompress_page(
    compressed_data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: i32,
) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(compressed_data.to_vec()),
        CompressionCodec::Snappy => Err(ParquetDemoError::Decode(
            "Snappy decompression not implemented".into(),
        )),
        CompressionCodec::Gzip => Err(ParquetDemoError::Decode(
            "Gzip decompression not implemented".into(),
        )),
        CompressionCodec::Lzo => Err(ParquetDemoError::Decode(
            "LZO decompression not implemented".into(),
        )),
        CompressionCodec::Brotli => Err(ParquetDemoError::Decode(
            "Brotli decompression not implemented".into(),
        )),
        CompressionCodec::Lz4 => Err(ParquetDemoError::Decode(
            "LZ4 decompression not implemented".into(),
        )),
        CompressionCodec::Zstd => {
            let decompressed = zstd::decode_all(compressed_data)?;
            if decompressed.len() != uncompressed_size as usize {
                return Err(ParquetDemoError::Decode(format!(
                    "Decompressed size mismatch: expected {}, got {}",
                    uncompressed_size,
                    decompressed.len()
                )));
            }
            Ok(decompressed)
        }
    }
}

/// Decode Plain-encoded Int64 values
pub fn decode_plain_int64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    let mut cursor = Cursor::new(data);
    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        let value = cursor.read_i64::<LittleEndian>()?;
        values.push(value);
    }

    Ok(values)
}

/// Decode RLE bit-packed hybrid and also return how many bytes were consumed
/// Simplified: optimized for bit_width <= 1 (definition levels in common cases)
pub fn decode_rle_bit_packed_hybrid_with_consumed(
    data: &[u8],
    bit_width: u8,
    max_values: usize,
) -> Result<(Vec<u8>, usize)> {
    let mut cursor = Cursor::new(data);
    let mut result = Vec::new();

    while result.len() < max_values {
        let header = read_varint(&mut cursor)?;

        if header & 1 == 0 {
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

/// Decode dictionary-encoded values (returns dictionary indices)
pub fn decode_rle_dictionary(data: &[u8], count: usize, bit_width: u8) -> Result<Vec<u32>> {
    let mut cursor = Cursor::new(data);
    let mut result = Vec::new();

    while result.len() < count {
        let header = match read_varint(&mut cursor) {
            Ok(h) => h,
            Err(_) => break,
        };

        if header & 1 == 0 {
            let run_length = (header >> 1) as usize;

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
                return Err(ParquetDemoError::Decode(
                    "Dictionary bit width > 8 not supported".into(),
                ));
            };

            for _ in 0..run_length {
                result.push(value);
                if result.len() >= count {
                    break;
                }
            }
        } else {
            let count_groups = (header >> 1) as usize;
            let values_to_read = std::cmp::min(count_groups * 8, count - result.len());
            let bytes_needed = (values_to_read * bit_width as usize).div_ceil(8);

            let mut packed_bytes = vec![0u8; bytes_needed];
            cursor.read_exact(&mut packed_bytes)?;

            let mut bit_offset = 0;
            for _ in 0..values_to_read {
                if result.len() >= count {
                    break;
                }

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

/// Decode page data based on type and encoding
pub fn decode_page_data(
    page_data: &[u8],
    page_type: Type,
    encoding: Encoding,
    num_values: i32,
) -> Result<PageValues> {
    match (page_type, encoding) {
        (Type::Int64, Encoding::Plain) => {
            let values = decode_plain_int64(page_data, num_values as usize)?;
            Ok(PageValues::Int64(values))
        }
        (Type::Int64, Encoding::RleDictionary) => {
            if page_data.is_empty() {
                return Ok(PageValues::DictionaryIndices(Vec::new()));
            }
            let bit_width = page_data[0];
            let indices = decode_rle_dictionary(&page_data[1..], num_values as usize, bit_width)?;
            Ok(PageValues::DictionaryIndices(indices))
        }
        _ => Err(ParquetDemoError::Decode(format!(
            "Decoding not implemented for type {:?} with encoding {:?}",
            page_type, encoding
        ))),
    }
}

fn read_varint(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
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

fn read_bits(cursor: &mut Cursor<&[u8]>, bit_width: u8) -> Result<u8> {
    if bit_width == 0 {
        return Ok(0);
    }

    if bit_width == 1 {
        let byte = cursor.read_u8()?;
        Ok(byte)
    } else {
        let byte = cursor.read_u8()?;
        Ok(byte & ((1 << bit_width) - 1))
    }
}
