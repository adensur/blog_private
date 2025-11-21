use std::{fmt, io};

pub type Result<T> = std::result::Result<T, ParquetDemoError>;

#[derive(Debug)]
pub enum ParquetDemoError {
    Io(io::Error),
    Thrift(thrift::Error),
    Serde(serde_json::Error),
    ArrowSchema(String),
    InvalidMagic {
        kind: &'static str,
        expected: [u8; 4],
        actual: [u8; 4],
    },
    InvalidFooterLength(i32),
    MissingColumnMeta {
        column_idx: usize,
    },
    Decode(String),
    Message(String),
}

impl fmt::Display for ParquetDemoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParquetDemoError::Io(e) => write!(f, "I/O error: {}", e),
            ParquetDemoError::Thrift(e) => write!(f, "Thrift error: {}", e),
            ParquetDemoError::Serde(e) => write!(f, "Serde error: {}", e),
            ParquetDemoError::ArrowSchema(msg) => write!(f, "Arrow schema error: {}", msg),
            ParquetDemoError::InvalidMagic {
                kind,
                expected,
                actual,
            } => write!(
                f,
                "Invalid Parquet {} magic: expected {:?}, found {:?}",
                kind, expected, actual
            ),
            ParquetDemoError::InvalidFooterLength(len) => {
                write!(f, "Invalid footer length: {}", len)
            }
            ParquetDemoError::MissingColumnMeta { column_idx } => {
                write!(f, "Missing column metadata for column index {}", column_idx)
            }
            ParquetDemoError::Decode(msg) => write!(f, "Decoding error: {}", msg),
            ParquetDemoError::Message(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for ParquetDemoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParquetDemoError::Io(e) => Some(e),
            ParquetDemoError::Thrift(e) => Some(e),
            ParquetDemoError::Serde(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ParquetDemoError {
    fn from(value: io::Error) -> Self {
        ParquetDemoError::Io(value)
    }
}

impl From<thrift::Error> for ParquetDemoError {
    fn from(value: thrift::Error) -> Self {
        ParquetDemoError::Thrift(value)
    }
}

impl From<serde_json::Error> for ParquetDemoError {
    fn from(value: serde_json::Error) -> Self {
        ParquetDemoError::Serde(value)
    }
}
