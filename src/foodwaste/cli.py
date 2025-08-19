"""
Command Line Interface for FoodWaste semantic segmentation training
"""

import typer
from pathlib import Path
from typing import Optional
import yaml
import traceback
from datetime import datetime

from .config import MainConfig, ROOT
from .train import runner
from .utils import setup_logging

app = typer.Typer(
    name="foodwaste",
    help="FoodWaste semantic segmentation training CLI",
    add_completion=False,
    rich_markup_mode="rich"
)


def load_config(config_path: Path) -> MainConfig:
    """Load configuration from YAML file"""
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist", err=True)
        raise typer.Exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return MainConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option(
        "configs/train.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    override_config: Optional[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Override configuration values (format: key=value,key2=value2)",
    ),
):
    """
    Train a FoodWaste semantic segmentation model using PyTorch Lightning
    """
    typer.echo("🚀 Starting FoodWaste training...")
    
    # Load configuration
    typer.echo(f"📋 Loading configuration from {config}")
    main_config = load_config(config)

    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    setup_logging(log_dir=ROOT / "logs", log_file=log_file)

       
    # Display configuration summary
    typer.echo("\n📊 Configuration Summary:")
    typer.echo(f"   Model: {main_config.model.model_name}")
    typer.echo(f"   Epochs: {main_config.training.num_epochs}")
    typer.echo(f"   Batch Size: {main_config.training.batch_size}")
    typer.echo(f"   Learning Rate: {main_config.training.learning_rate}")
    typer.echo(f"   Device: {main_config.training.device}")
    typer.echo(f"   Save Directory: {main_config.logging.save_dir}")
    
    # Confirm before starting
    #if not typer.confirm("\nProceed with training?"):
    #    typer.echo("Training cancelled.")
    #    raise typer.Exit(0)
    
    try:
        # Start training
        typer.echo("\n🎯 Starting training...")
        runner(main_config)
        typer.echo("✅ Training completed successfully!")
    except Exception as e:
        typer.echo(f"❌ Training failed: {traceback.format_exc()}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Option(
        "configs/train.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """
    Validate configuration file without starting training
    """
    typer.echo("🔍 Validating configuration...")
    
    raise NotImplementedError("Validation command not implemented")


@app.command()
def info():
    """
    Display information about the FoodWaste project
    """
    typer.echo("🍽️  FoodWaste Semantic Segmentation")
    typer.echo("=" * 40)
    typer.echo("A PyTorch Lightning-based framework for training")
    typer.echo("semantic segmentation models on food waste datasets.")
    typer.echo("\nAvailable commands:")
    typer.echo("  train     - Train a model")
    typer.echo("  validate  - Validate configuration")
    typer.echo("  info      - Show this information")
    typer.echo("\nFor more information, visit:")
    typer.echo("  https://github.com/your-repo/foodwaste")

