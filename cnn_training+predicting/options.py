
class Configuration:
    """
    Configuration options for the training
    """
    def __init__(self,
                 batch_size,
                 number_of_epochs,
                 this_run_folder,
                 start_epoch,
                 experiment_name,
                 quality_factor,
                 device,
                 img_path,
                 weight,
                 height,
                 ):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.this_run_folder = this_run_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name
        self.quality_factor = quality_factor
        self.device = device
        self.img_path = img_path
        self.weight = weight
        self.height = height
