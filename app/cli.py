from pathlib import Path

import click

from .main import compute_analysis


@click.group()
def atlas():
    """Atlas CLI tool for dataset analysis and management."""
    pass


@atlas.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset directory",
)
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    help="Path to the COCO annotations file (optional)",
)
def analyse(dataset_dir: str, annotations_path: str | None):
    """Analyse a dataset and generate analysis files."""
    click.echo(f"Analysing dataset in {dataset_dir}")
    if annotations_path:
        click.echo(f"Using annotations from {annotations_path}")

    # Convert paths to absolute paths
    dataset_dir = str(Path(dataset_dir).resolve())
    if annotations_path:
        annotations_path = str(Path(annotations_path).resolve())

    # Call the analysis function
    compute_analysis(dataset_dir, annotations_path)
    click.echo(
        "Analysis complete! Check image_df.csv and annotation_df.csv for results."
    )


if __name__ == "__main__":
    atlas()
