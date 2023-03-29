import json
import os

# shutil module provides a higher level interface when copying files or directories
import shutil
import torch
import torch.optim as optim
from networks.EarthNetComplex import EarthNetComplex  # import EarthNetComplex neural network model

from data.data_manager import DataManager  # import DataManager class to handle dataset loading and preprocessing
from trainer import Trainer  # import Trainer class to handle model training
from utils.data_logs import save_logs_about  # import function to save logs
import utils.losses as loss_functions  # import module with different loss functions


def main():
    # Load configuration from config.json file
    config = json.load(open('./config.json'))

    # Set device to use GPU if available, otherwise use CPU
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Create a new directory to save the experiment files
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    # Create an instance of the EarthNetComplex model and move it to the selected device
    model = EarthNetComplex().to(config['device'])

    # Initialize model weights using EarthNetComplex.init_weights() method
    model.apply(EarthNetComplex.init_weights)

    # Save information about experiment to log file
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    # Copy the model file to the experiment directory
    shutil.copy(model.get_path(), os.path.join(config['exp_path'], config['exp_name']))

    # Get the criterion function based on the selected loss function in the configuration file
    criterion = getattr(loss_functions, config['loss_function'])

    # Initialize an optimizer using Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Initialize a learning rate scheduler using StepLR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'], last_epoch=-1)

    # Create an instance of the DataManager class and get data loaders for train, validation, and test datasets
    data_manager = DataManager(config)
    train_loader, validation_loader, test_loader = data_manager.get_train_eval_test_dataloaders()

    # Create an instance of the Trainer class and train the model
    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)
    trainer.train()

    # Test the trained model on the test dataset
    trainer.test_net(test_loader)


def test_net():
    # Function made only to test a pretrained network.

    # Load configuration from config.json file
    config = json.load(open('./config.json'))

    # Set device to use GPU if available, otherwise use CPU
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create an instance of the EarthNetComplex model and move it to the selected device
    model = EarthNetComplex().to(config['device'])

    # Load the trained weights of the model from a saved checkpoint file in the experiment directory
    checkpoint = torch.load(os.path.join(config['exp_path'], config['exp_name'], 'latest_checkpoint.pkl'),
                            map_location=config['device'])
    model.load_state_dict(checkpoint['model_weights'])

    # Get the criterion function based on the selected loss function in the configuration file
    criterion = getattr(loss_functions, config['loss_function'])

    # Create an instance of the DataManager class and get data loaders for train, validation, and
    data_manager = DataManager(config)
    _, _, test_loader = data_manager.get_train_eval_test_dataloaders()

    trainer = Trainer(model, None, None, criterion, None, None, config)
    trainer.test_net(test_loader)


if __name__ == "__main__":
    main()
