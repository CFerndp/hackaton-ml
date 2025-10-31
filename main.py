import argparse
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

def preprocess_command(args):
    """Execute preprocessing pipeline."""
    print("🔧 Starting MIDUS data preprocessing...")
    
    try:
        from preprocessing import preprocess_midus_data
        
        # Default data path if not provided (relative to project root)
        if not args.data_path:
            args.data_path = "data/25. 0504 Harvard dataverse. Stree, Inflammation, Cognition. sav.tab"
        
        # Convert to Path object for better handling
        data_path = Path(args.data_path)
        
        # If path is not absolute, make it relative to project root
        if not data_path.is_absolute():
            data_path = project_root / args.data_path
        
        # Check if file exists
        if not data_path.exists():
            print(f"❌ Error: Data file not found at {data_path}")
            print("Available files in data/:")
            data_dir = project_root / "data"
            if data_dir.exists():
                for file in data_dir.iterdir():
                    if file.is_file():
                        print(f"   • {file.name}")
            else:
                print("   (data/ directory doesn't exist)")
            return False
        
        # Create necessary directories
        (project_root / "data").mkdir(exist_ok=True)
        (project_root / "results").mkdir(exist_ok=True)
        
        # Run preprocessing
        preprocessor, processed_data = preprocess_midus_data(
            data_path=str(data_path),
            outlier_strategy=args.outlier_strategy,
            scaling_method=args.scaling_method
        )
        
        # Save processed data
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = project_root / args.output
        else:
            output_path = project_root / "data" / "midus_processed.csv"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_csv(output_path, index=False)
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📊 Processed data shape: {processed_data.shape}")
        print(f"💾 Data saved to: {output_path.relative_to(project_root)}")
        
        # Show summary statistics
        if args.verbose:
            print(f"\n📈 Summary Statistics:")
            print(f"   • Original features: {preprocessor.original_df.shape[1]}")
            print(f"   • Final features: {processed_data.shape[1]}")
            print(f"   • New derived features: {processed_data.shape[1] - preprocessor.original_df.shape[1]}")
            
            if 'cognitive_profile' in processed_data.columns:
                print(f"\n🧠 Cognitive Profile Distribution:")
                profile_counts = processed_data['cognitive_profile'].value_counts()
                for profile, count in profile_counts.items():
                    pct = (count / len(processed_data)) * 100
                    print(f"   • {profile}: {count} ({pct:.1f}%)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing preprocessing module: {e}")
        print("Make sure the preprocessing.py file exists in the src/ directory.")
        return False
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False


def train_command(args):
    """Execute training pipeline."""
    print("🚀 Starting MIDUS model training...")
    
    try:
        from train import train
        
        # Check if processed data exists
        if args.data_path:
            data_path = Path(args.data_path)
            if not data_path.is_absolute():
                data_path = project_root / args.data_path
        else:
            data_path = project_root / "data" / "midus_processed.csv"
            
        if not data_path.is_file():
            print(f"❌ Error: Processed data not found at {data_path.relative_to(project_root)}")
            print("Please run preprocessing first:")
            print("   uv run python main.py preprocess")
            return False
        
        # Create models directory
        (project_root / "models").mkdir(exist_ok=True)
        
        # Run training 
        result = train()
        
        if result:
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing train module: {e}")
        print("Make sure the train.py file exists in the src/ directory.")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False


def evaluate_command(args):
    """Execute model evaluation."""
    print("📊 Starting model evaluation...")
    
    try:
        # Import evaluation functions (you'll need to create this)
        # from evaluate import evaluate_model
        
        print("🔍 Model evaluation functionality coming soon...")
        print("This will include:")
        print("   • Model performance metrics")
        print("   • Feature importance analysis")
        print("   • Cross-validation results")
        print("   • Confusion matrices")
        print("   • ROC curves")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False


def predict_command(args):
    """Execute predictions on new data."""
    print("🔮 Starting prediction pipeline...")
    
    try:
        print("🔍 Prediction functionality coming soon...")
        print("This will include:")
        print("   • Load trained models")
        print("   • Preprocess new data")
        print("   • Generate predictions")
        print("   • Export results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False


def analyze_command(args):
    """Execute data analysis and visualization."""
    print("📈 Starting data analysis...")
    
    try:
        from preprocessing import MIDUSPreprocessor
        
        # Default data path if not provided
        if args.data_path:
            data_path = Path(args.data_path)
            if not data_path.is_absolute():
                data_path = project_root / args.data_path
        else:
            data_path = project_root / "data" / "25. 0504 Harvard dataverse. Stree, Inflammation, Cognition. sav.tab"
        
        if not data_path.exists():
            print(f"❌ Error: Data file not found at {data_path.relative_to(project_root)}")
            print("Available files in data/:")
            data_dir = project_root / "data"
            if data_dir.exists():
                for file in data_dir.iterdir():
                    if file.is_file():
                        print(f"   • {file.name}")
            return False
        
        # Create results directory
        (project_root / "results").mkdir(exist_ok=True)
        
        # Load data and create basic analysis
        preprocessor = MIDUSPreprocessor(data_path=str(data_path))
        preprocessor.explore_data(save_plots=True)
        
        print("✅ Data analysis completed!")
        print("📊 Check the generated plots in results/:")
        print("   • midus_data_exploration.png")
        print("   • midus_correlation_matrix.png") 
        print("   • midus_age_cognition.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False


def list_command(args):
    """List available data files and results."""
    print("📁 Project contents:")
    
    # Data files
    data_dir = project_root / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        print("\n📊 Data files:")
        for file in sorted(data_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size / (1024*1024)  # MB
                print(f"   • {file.name} ({size:.1f} MB)")
    else:
        print("\n📊 Data files: (none)")
    
    # Results
    results_dir = project_root / "results"
    if results_dir.exists() and any(results_dir.iterdir()):
        print("\n📈 Results:")
        for file in sorted(results_dir.iterdir()):
            if file.is_file():
                print(f"   • {file.name}")
    else:
        print("\n📈 Results: (none)")
    
    # Models
    models_dir = project_root / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        print("\n🤖 Models:")
        for file in sorted(models_dir.iterdir()):
            if file.is_file():
                print(f"   • {file.name}")
    else:
        print("\n🤖 Models: (none)")
    
    return True


def clean_command(args):
    """Clean generated files."""
    print("🧹 Cleaning generated files...")
    
    cleaned_files = 0
    
    # Clean results
    if args.results or args.all:
        results_dir = project_root / "results"
        if results_dir.exists():
            for file in results_dir.iterdir():
                if file.is_file():
                    file.unlink()
                    cleaned_files += 1
            print(f"✓ Cleaned {cleaned_files} result files")
    
    # Clean processed data
    if args.data or args.all:
        processed_files = [
            "data/midus_processed.csv",
            "data/midus_processed_data.csv"
        ]
        for file_path in processed_files:
            file = project_root / file_path
            if file.exists():
                file.unlink()
                cleaned_files += 1
        print(f"✓ Cleaned processed data files")
    
    # Clean models
    if args.models or args.all:
        models_dir = project_root / "models"
        if models_dir.exists():
            model_files = 0
            for file in models_dir.iterdir():
                if file.is_file():
                    file.unlink()
                    model_files += 1
            if model_files > 0:
                print(f"✓ Cleaned {model_files} model files")
    
    if cleaned_files == 0:
        print("No files to clean.")
    else:
        print(f"✅ Cleaned {cleaned_files} files total")
    
    return True


def main():
    """Main CLI interface."""
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="🧠 MIDUS Cognitive Health & Biomarker Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  uv run python main.py preprocess --data-path data/midus_raw.tab
  
  # Train classification model
  uv run python main.py train --target cognitive_profile --model random_forest
  
  # Analyze data with visualizations
  uv run python main.py analyze
  
  # List project contents
  uv run python main.py list
  
  # Clean generated files
  uv run python main.py clean --all

Dependencies are managed by uv. No setup required!
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess MIDUS data",
        description="Clean, transform, and prepare MIDUS data for analysis"
    )
    preprocess_parser.add_argument(
        "--data-path", "-d",
        help="Path to raw MIDUS data file (relative to project root)",
        default=None
    )
    preprocess_parser.add_argument(
        "--output", "-o",
        help="Output path for processed data (relative to project root)",
        default=None
    )
    preprocess_parser.add_argument(
        "--outlier-strategy",
        choices=["cap", "remove", "keep"],
        default="cap",
        help="Strategy for handling outliers (default: cap)"
    )
    preprocess_parser.add_argument(
        "--scaling-method",
        choices=["robust", "standard", "none"],
        default="robust", 
        help="Feature scaling method (default: robust)"
    )
    preprocess_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML models",
        description="Train machine learning models on preprocessed MIDUS data"
    )
    train_parser.add_argument(
        "--data-path", "-d",
        help="Path to preprocessed data (relative to project root)",
        default=None
    )
    train_parser.add_argument(
        "--target", "-t",
        help="Target variable for prediction",
        default="cognitive_profile"
    )
    train_parser.add_argument(
        "--model", "-m",
        choices=["random_forest", "gradient_boosting", "svm", "logistic"],
        default="random_forest",
        help="Model type to train (default: random_forest)"
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size as fraction (default: 0.2)"
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    train_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    # Evaluation command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained models",
        description="Evaluate model performance with various metrics"
    )
    evaluate_parser.add_argument(
        "--model-path",
        help="Path to trained model file (relative to project root)"
    )
    evaluate_parser.add_argument(
        "--data-path",
        help="Path to test data (relative to project root)"
    )
    
    # Prediction command  
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Make predictions on new data",
        description="Use trained models to make predictions"
    )
    predict_parser.add_argument(
        "--model-path",
        help="Path to trained model file (relative to project root)"
    )
    predict_parser.add_argument(
        "--data-path",
        help="Path to data for prediction (relative to project root)"
    )
    predict_parser.add_argument(
        "--output",
        help="Output path for predictions (relative to project root)"
    )
    
    # Analysis command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze and visualize data", 
        description="Generate exploratory data analysis and visualizations"
    )
    analyze_parser.add_argument(
        "--data-path", "-d",
        help="Path to data file (relative to project root)",
        default=None
    )
    
    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List project contents",
        description="Show available data files, results, and models"
    )
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean generated files",
        description="Remove generated files to free up space"
    )
    clean_parser.add_argument(
        "--results", "-r",
        action="store_true",
        help="Clean result files (plots, etc.)"
    )
    clean_parser.add_argument(
        "--data", "-d",
        action="store_true",
        help="Clean processed data files"
    )
    clean_parser.add_argument(
        "--models", "-m",
        action="store_true",
        help="Clean trained model files"
    )
    clean_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Clean all generated files"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Print header
    print("=" * 60)
    print("🧠 MIDUS Cognitive Health & Biomarker Analysis Tool")
    print("=" * 60)
    
    # Execute command
    success = False
    
    if args.command == "preprocess":
        success = preprocess_command(args)
    elif args.command == "train":
        success = train_command(args)
    elif args.command == "evaluate":
        success = evaluate_command(args)
    elif args.command == "predict":
        success = predict_command(args)
    elif args.command == "analyze":
        success = analyze_command(args)
    elif args.command == "list":
        success = list_command(args)
    elif args.command == "clean":
        success = clean_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
    
    # Exit with appropriate code
    if success:
        print("\n🎉 Command completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Command failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
