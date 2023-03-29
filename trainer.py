'''During training, the Trainer object iterates over the training data in batches and applies the forward and backward passes of the network using the optimizer and loss function to compute and update the gradients. The eval_net method is called after a specified number of epochs to evaluate the performance of the network on the validation data. The train method handles the overall training and evaluation process, as well as saving the network state after each epoch and at regular intervals. The learning rate is adjusted using the learning rate scheduler provided during initialization.'''

import torch
import numpy as np
from utils.stats_manager import StatsManager
from utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    #__init__: This method initializes the Trainer object with the provided network, data loaders, loss function, optimizer, learning rate scheduler, and configuration settings.
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, config):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.best_metric = 0.0
    #train_epoch: This method trains the network for one epoch using the training data and updates the weights using the backpropagation algorithm.
    def train_epoch(self, epoch):
        # Train the neural network model for one epoch.
        running_loss = []
        self.network.train()
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(self.train_dataloader, 0):
            # Get the input data and labels, move them to the appropriate device and data type.
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            # Reset the gradients of the optimizer.
            self.optimizer.zero_grad()
            # Forward the input data to the neural network model to get the output predictions.
            pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            # Compute the loss between the output predictions and the ground truth labels.
            loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                  labels_depth, labels_distance, labels_magnitude)
            # Compute the gradients of the loss with respect to the model parameters.
            loss.backward()
            # Update the model parameters using the optimizer.
            self.optimizer.step()

            # Add the loss value to a running list of losses.
            running_loss.append(loss.item())
            # Print the average training loss every self.config['print_loss'] iterations.
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                # Save the training loss to a log file.
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')
                running_loss = []
    #eval_net: This method evaluates the network on the validation data and calculates the evaluation loss, as well as the mean depth, distance, and magnitude errors using a StatsManager object.
    def eval_net(self, epoch):
        # Initialize empty lists to store predicted values and ground truth labels
        stats_pred_depth = []
        stats_pred_distance = []
        stats_pred_magnitude = []

        stats_lbl_depth = []
        stats_lbl_distance = []
        stats_lbl_magnitude = []

        # Initialize a variable to store the running evaluation loss
        running_eval_loss = 0.0

        # Set the network to evaluation mode (i.e., turn off gradients and batch normalization)
        self.network.eval()

        # Iterate over the evaluation dataloader
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(self.eval_dataloader, 0):
            # Move the inputs and labels to the specified device and convert them to float tensors
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            # Turn off gradients and make predictions using the network
            with torch.no_grad():
                pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            # Calculate the evaluation loss
            eval_loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                       labels_depth, labels_distance, labels_magnitude)

            # Add the current evaluation loss to the running total
            running_eval_loss += eval_loss.item()

            # Store the predicted values and ground truth labels for later analysis
            stats_pred_depth.append(pred_depth.detach().cpu().numpy())
            stats_pred_distance.append(pred_distance.detach().cpu().numpy())
            stats_pred_magnitude.append(pred_magnitude.detach().cpu().numpy())

            stats_lbl_depth.append(labels_depth.detach().cpu().numpy())
            stats_lbl_distance.append(labels_distance.detach().cpu().numpy())
            stats_lbl_magnitude.append(labels_magnitude.detach().cpu().numpy())

        # Calculate the mean error for each prediction type (depth, distance, magnitude)
        mean_depth_err, mean_distance_err, mean_magnitude_err = \
            self.stats_manager.get_stats(pred_depth=stats_pred_depth, pred_distance=stats_pred_distance, pred_magnitude=stats_pred_magnitude,
                                         lbl_depth=stats_lbl_depth, lbl_distance=stats_lbl_distance, lbl_magnitude=stats_lbl_magnitude)

        # Calculate the average evaluation loss across all batches
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        # Print the evaluation metrics
        print(f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, '
              f'mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}')

        # Save the evaluation metrics to a log file
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, '
                       f'mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}')

        # Check if the current model has achieved a new best metric and save the network state if it has
        if self.best_metric < mean_magnitude_err:
            self.best_metric = mean_magnitude_err
            self.save_net_state(None, best=True)

    #train: This method trains the network for the specified number of epochs and saves the model state after each epoch and at regular intervals. The learning rate is adjusted after each epoch using the learning rate scheduler.
    def train(self):
        # Resume training if specified in the configuration
        if self.config['resume_training'] is True:
            # Load the latest saved checkpoint of the model and optimizer
            checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'])
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Train for the specified number of epochs
        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            # Train for one epoch
            self.train_epoch(i)
            # Save the latest checkpoint of the model and optimizer
            self.save_net_state(i, latest=True)

            # Evaluate the model on the validation set at a specified interval
            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            # Save the model at a specified interval
            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            # Update the learning rate schedule
            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=False):
        # Save the latest checkpoint of the model and optimizer
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        # Save the best model based on a specified metric
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)
        # Save the model at a specified epoch
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pkl')
            torch.save(self.network, path_to_save)

    def test_net(self, test_dataloader):
        # Initialize empty lists to hold predictions and labels for each sample
        stats_pred_depth = []
        stats_pred_distance = []
        stats_pred_magnitude = []

        stats_lbl_depth = []
        stats_lbl_distance = []
        stats_lbl_magnitude = []

        running_loss = 0.0
        # Set the model to evaluation mode
        self.network.eval()
        # Iterate over the test dataset
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(test_dataloader, 0):
            # Move data to the device specified in the configuration
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            with torch.no_grad():
                # Compute the model's predictions for the current batch
                pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            # Compute the evaluation loss for the current batch
            eval_loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                       labels_depth, labels_distance, labels_magnitude)
            running_loss += eval_loss.item()

            stats_pred_depth.append(pred_depth.detach().cpu().numpy())
            stats_pred_distance.append(pred_distance.detach().cpu().numpy())
            stats_pred_magnitude.append(pred_magnitude.detach().cpu().numpy())

            stats_lbl_depth.append(labels_depth.detach().cpu().numpy())
            stats_lbl_distance.append(labels_distance.detach().cpu().numpy())
            stats_lbl_magnitude.append(labels_magnitude.detach().cpu().numpy())

        mean_depth_err, mean_distance_err, mean_magnitude_err = \
            self.stats_manager.get_stats(pred_depth=stats_pred_depth, pred_distance=stats_pred_distance,
                                         pred_magnitude=stats_pred_magnitude,
                                         lbl_depth=stats_lbl_depth, lbl_distance=stats_lbl_distance,
                                         lbl_magnitude=stats_lbl_magnitude)
        running_eval_loss = running_loss / len(test_dataloader)

        stats_description = f'### Test loss = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, \
                              mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}'

        print(stats_description)
        history = open(os.path.join(self.config['exp_path'], self.config['exp_name'], '__testStats__.txt'), "a")
        history.write(stats_description)
        history.close()
