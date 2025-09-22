
import argparse
from src.data_curation.curate import curate_data
from src.classical_ml.run import run_classical_ml_experiment

def main():
    """
    Main function to run the pipeline.
    """
    parser = argparse.ArgumentParser(description='Run the predictive modeling pipeline.')
    parser.add_argument('action', choices=['curate_data', 'run_experiment'], help='Action to perform')
    parser.add_argument('--model', default='classical_ml', help='Model to run the experiment for')

    args = parser.parse_args()

    if args.action == 'curate_data':
        curate_data()
    elif args.action == 'run_experiment':
        if args.model == 'classical_ml':
            run_classical_ml_experiment()
        else:
            print(f"Experiment for model '{args.model}' is not implemented yet.")

if __name__ == '__main__':
    main()
