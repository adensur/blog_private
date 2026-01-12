use parquet_writer::{DemoPrinter, ParquetFile, Result};

fn main() -> Result<()> {
    let file_path = "data.parquet";

    println!("Reading Parquet file: {}", file_path);
    let mut parquet = ParquetFile::open(file_path)?;
    let printer = DemoPrinter::default();

    printer.print_metadata(parquet.metadata())?;
    printer.print_arrow_schema(parquet.metadata())?;
    printer.print_row_groups(parquet.metadata())?;
    printer.print_data_pages(&mut parquet)?;

    Ok(())
}
