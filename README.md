# Installation

Install webots from https://www.cyberbotics.com/

Clone this repo.

Read the report and presentation for more details.

## Important files

train.py - used to train the model

dataset_gen.py - used to generate the csv for training the model

dataset.py - contains dataset class and loader, used in train.py

config.json - various hyperparams

Sim_ATAV/vehicle_control/automated_driving_with_fusion2.py - file used for controlling speed, saving images, running models etc

src - contains things that aren't used, but may help in future work

src/wbo_files - important wbo and PROTO files which are needed in webots like our tesla model etc

Webots_Projects/controllers/vehicle_controller - used while running the car (need to change the ML model inside) for testing

Webots_Projects/worlds - contains scenarios to generate dataset, and test the models

PROTO file - $INST_DIR/projects/vehicles/protos/tesla


# ignore

wiki_files
tests