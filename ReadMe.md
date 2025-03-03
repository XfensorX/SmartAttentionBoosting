> ⚠️ **Note:** 
> Pressure makes diamonds, but it doesn’t always make perfectly structured code.
> A full refactoring is still on the to-do list—please note that this project doesn’t reflect my views on coding standards.

---
# Code base for "Smart Federated Averaging"

This code base implements the 'Smart Federated Averaging' technique, which can be used in combination with an adapted scaled-dot-product attention as "Smart Attention Layer", which can be stacked to form very effective "Smart Attention Boosting".

### Install the package:

-   go to root dir of the project
-   then:

```shell
pip install -e .
pip install -f requirements.txt
```

-> Python version 3.10 was used to create this project

### Run tests with:

```shell
python -m pytest utils/tests/
```

## Project Structure:

The project is structured into several subfolders:

-   [experiments](experiments): contains subfolders for each dataset on which is trained.

    -   [adult](experiments/adult/main.py) is the main script for running experiments on the adult dataset. One can easily configurate a .yaml - file based on the once already located there, stating hyperparameters:

    ```shell
    python main.py --config my_config.yaml --device <cpu|mps|cuda|...>
    ```

    -   Please note, that the usage of a device other than the cpu is not well tested and generally worse, as the concatenation of bias features ([see here](models/multi_output_net.py): line 175) is slow

    -   [artifical 1D](experiments/artificial_1D_linear/): Notebooks are included which train different models on this artifical dataset. However they are partly outdated and should be replaced by a script using configuration files.

-   [training](training): contains training scripts with training functions for each of the models

-   [models](models): Contains the Pytorch Modules for the different models used.

    -   [Multi Output Net](models/multi_output_net.py) The main "Smart federated averaging model" with all subfunctions that handle weight merging, etc.
    -   [Smart Attention Layer](models/smart_attention_layer.py) Uses several of the "Smart federated averaging model" to bring attention based client selection to federated learning
    -   [Smart Attention Boosting](models/smart_attention_boosting.py) Uses several "Smart Attention Layers" to train residuals of former outputs. Decreases communication overhead in federated learning by only training the last layer at a time.

    -   [Smart Average Layer](models/smart_average_layer.py): Uses the "Smart federated averaging model" with equal client weightsm in order to provide a training option without using attention, not well tested
    -   [Smart Average Boosting](models/smart_average_boosting.py): Uses the "Smart Average Layer" for boosting. not well tested.

-   [data](data): contains subfolders for different datasets used and scripts to prepare them for usage in experiments. The ones used until now are:

    -   [adult](data\adult\data.py) the adult dataset from [The UC Irvine](https://archive.ics.uci.edu/dataset/2/adult)
    -   [artifical 1D](data/artificial_1D_linear/data.py): a custom dataset for training purposes

-   [logs](logs): contains subfolder for each dataset, following subfolders for different models, logs are saved using tensorboard writers, they can therefore be read using, e.g.:

```shell
tensorboard --logdir logs/adult
```

-   [utils](utils): Contains several scripts with helper functions throughout the code

    -   [self_learning.py](utils/self_learning.py) contains the key components for the "Smart Federated Averaging" of weights.
    -   [tests](utils/tests/) contains some test functions in order to test some of the key features of methods throughout this project

-   [theory](theory): notebooks used to work on some theoretical components, do not have further use than that

-   [report](report): contains the bibliography used in development

-   [docs](docs): contains documentation of commonly used libraries for offline usage. They are downloaded from the web.
