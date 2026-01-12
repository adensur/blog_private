use byteorder::{LittleEndian, ReadBytesExt};
use parquet_format::FileMetaData;
use std::{
    fs::File,
    io::{Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};
use thrift::protocol::TCompactInputProtocol;

use crate::errors::{ParquetDemoError, Result};

const PARQUET_MAGIC: [u8; 4] = *b"PAR1";

pub struct ParquetFile {
    path: PathBuf,
    file: File,
    metadata: Arc<FileMetaData>,
}

impl ParquetFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&path_buf)?;

        let header_magic = validate_magic(&mut file, SeekFrom::Start(0), "header")?;
        println!(
            "✓ Valid magic header: {}",
            String::from_utf8_lossy(&header_magic)
        );

        let footer_magic = validate_magic(&mut file, SeekFrom::End(-4), "footer")?;
        println!(
            "✓ Valid magic footer: {}",
            String::from_utf8_lossy(&footer_magic)
        );

        let footer_length = read_footer_length(&mut file)?;
        println!("✓ Footer length: {} bytes", footer_length);

        let metadata = Arc::new(read_file_metadata(&mut file, footer_length)?);

        file.seek(SeekFrom::Start(0))?;

        Ok(Self {
            path: path_buf,
            file,
            metadata,
        })
    }

    pub fn metadata(&self) -> &FileMetaData {
        self.metadata.as_ref()
    }

    pub fn metadata_arc(&self) -> Arc<FileMetaData> {
        Arc::clone(&self.metadata)
    }

    pub fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn validate_magic(file: &mut File, seek: SeekFrom, kind: &'static str) -> Result<[u8; 4]> {
    file.seek(seek)?;
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;

    if buffer != PARQUET_MAGIC {
        return Err(ParquetDemoError::InvalidMagic {
            kind,
            expected: PARQUET_MAGIC,
            actual: buffer,
        });
    }

    Ok(buffer)
}

fn read_footer_length(file: &mut File) -> Result<i32> {
    file.seek(SeekFrom::End(-8))?;
    let footer_length = file.read_i32::<LittleEndian>()?;

    if footer_length <= 0 {
        return Err(ParquetDemoError::InvalidFooterLength(footer_length));
    }

    Ok(footer_length)
}

fn read_file_metadata(file: &mut File, footer_length: i32) -> Result<FileMetaData> {
    file.seek(SeekFrom::End(-8 - footer_length as i64))?;
    let mut metadata_bytes = vec![0u8; footer_length as usize];
    file.read_exact(&mut metadata_bytes)?;

    let mut cursor = Cursor::new(metadata_bytes);
    let mut protocol = TCompactInputProtocol::new(&mut cursor);
    let metadata = FileMetaData::read_from_in_protocol(&mut protocol)?;

    Ok(metadata)
}
