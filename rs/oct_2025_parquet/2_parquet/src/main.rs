use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use thrift::protocol::TCompactInputProtocol;
use parquet_format::FileMetaData;
use parquet_writer::{file_metadata_to_json, row_group_to_json};

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
        ).into());
    }
    println!("✓ Valid magic header: {}", String::from_utf8_lossy(&header_magic));

    // Step 2: Read and verify magic footer
    file.seek(SeekFrom::End(-4))?;
    let mut footer_magic = [0u8; 4];
    file.read_exact(&mut footer_magic)?;

    if &footer_magic != PARQUET_MAGIC {
        return Err(format!(
            "Invalid Parquet file: expected magic footer {:?}, got {:?}",
            PARQUET_MAGIC, footer_magic
        ).into());
    }
    println!("✓ Valid magic footer: {}", String::from_utf8_lossy(&footer_magic));

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
    let mut protocol = TCompactInputProtocol::new(&mut cursor);
    let file_metadata = FileMetaData::read_from_in_protocol(&mut protocol)?;

    // Step 6: Print FileMetaData
    println!("\n{}", "=".repeat(80));
    println!("FILE METADATA");
    println!("{}", "=".repeat(80));
    let json_metadata = file_metadata_to_json(&file_metadata);
    println!("{}", serde_json::to_string_pretty(&json_metadata)?);

    // Step 7: Print each RowGroup
    println!("\n{}", "=".repeat(80));
    println!("ROW GROUPS (Total: {})", file_metadata.row_groups.len());
    println!("{}", "=".repeat(80));

    for (i, row_group) in file_metadata.row_groups.iter().enumerate() {
        println!("\n--- Row Group {} ---", i);
        let json_rg = row_group_to_json(row_group, i);
        println!("{}", serde_json::to_string_pretty(&json_rg)?);
    }

    Ok(())
}
