import argparse
import main


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps_epoch',
        help='Steps per epoch',
        type=int,
        default=3
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs',
        type=int,
        default=2
    )
    parser.add_argument(
        '--batch_size',
        help='Batch size',
        type=int,
        default=2
    )
    parser.add_argument(
        '--data_path',
        help='Path to training/validation data',
        type=str
    )
    parser.add_argument(
        '--model',
        help='Model to be used.',
        type=str,
        default='model_1'
    )
    parser.add_argument(
        '--run_id',
        help='ID of the run.',
        type=str,
    )

    args = parser.parse_args()
    args.STEPS_EPOCHS = args.steps_epoch
    args.EPOCHS = args.epochs
    args.MODEL = args.model
    args.BATCH_SIZE = args.batch_size
    args.DATA_PATH = args.data_path
    args.RUN_ID = args.run_id
    return args


if __name__ == '__main__':
    args = arguments()
    main.model(args)
