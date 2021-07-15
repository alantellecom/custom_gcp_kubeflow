
"""KFP orchestrating BigQuery and Cloud AI Platform services."""

import os

from helper_components import evaluate_model
from helper_components import retrieve_best_run
from helper_components import prepoc_split_dataset
from jinja2 import Template
import kfp
from kfp.components import func_to_container_op
from kfp.dsl.types import Dict
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret

# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')

TRAINING_FILE_PATH = 'datasets/training/data.csv'
VALIDATION_FILE_PATH = 'datasets/validation/data.csv'
TESTING_FILE_PATH = 'datasets/testing/data.csv'

# Parameter defaults
SPLITS_DATASET_ID = 'splits'
HYPERTUNE_SETTINGS = """
{
    "hyperparameters":  {
        "goal": "MAXIMIZE",
        "maxTrials": 6,
        "maxParallelTrials": 3,
        "hyperparameterMetricTag": "accuracy",
        "enableTrialEarlyStopping": True,
        "params": [
            {
                "parameterName": "max_iter",
                "type": "DISCRETE",
                "discreteValues": [80, 100]
            },
            {
                "parameterName": "alpha",
                "type": "DOUBLE",
                "minValue": 0.001,
                "maxValue": 0.009,
                "scaleType": "UNIT_LINEAR_SCALE"
            }
        ]
    }
}
"""

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

prepoc_split_op = func_to_container_op(prepoc_split_dataset, base_image=BASE_IMAGE)
mlengine_train_op = component_store.load_component('ml_engine/train')
mlengine_deploy_op = component_store.load_component('ml_engine/deploy')
retrieve_best_run_op = func_to_container_op(
    retrieve_best_run, base_image=BASE_IMAGE)
evaluate_model_op = func_to_container_op(evaluate_model, base_image=BASE_IMAGE)


@kfp.dsl.pipeline(
    name='Iris Classifier Training',
    description='The pipeline training and deploying the Iris classifierpipeline_yaml'
)
def iris_train(project_id,
                    region,
                    gcs_root,
                    dataset_id,
                    evaluation_metric_name,
                    evaluation_metric_threshold,
                    model_id,
                    version_id,
                    replace_existing_version,
                    hypertune_settings=HYPERTUNE_SETTINGS,
                    dataset_location='US'):
    """Orchestrates training and deployment of an sklearn model."""

    splited_prepoc_dataset = prepoc_split_op(gcs_root)
   
    # Tune hyperparameters
    tune_args = [
        '--training_dataset_path',
        splited_prepoc_dataset.outputs['training_file_path'],
        '--validation_dataset_path',
        splited_prepoc_dataset.outputs['validation_file_path'], '--hptune', 'True'
    ]

    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir/hypertune',
                                kfp.dsl.RUN_ID_PLACEHOLDER)

    hypertune = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=tune_args,
        training_input=hypertune_settings)

    # Retrieve the best trial
    get_best_trial = retrieve_best_run_op(
            project_id, hypertune.outputs['job_id'])

    # Train the model on a combined training and validation datasets
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir', kfp.dsl.RUN_ID_PLACEHOLDER)

    train_args = [
        '--training_dataset_path',
        splited_prepoc_dataset.outputs['training_file_path'],
        '--validation_dataset_path',
        splited_prepoc_dataset.outputs['validation_file_path'], '--alpha',
        get_best_trial.outputs['alpha'], '--max_iter',
        get_best_trial.outputs['max_iter'], '--hptune', 'False'
    ]

    train_model = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=train_args)

    # Evaluate the model on the testing split
    eval_model = evaluate_model_op(
        testing_file_path=str(splited_prepoc_dataset.outputs['testing_file_path']),
        model_path=str(train_model.outputs['job_dir']),
        metric_name=evaluation_metric_name)

    # Deploy the model if the primary metric is better than threshold
    with kfp.dsl.Condition(eval_model.outputs['metric_value'] > evaluation_metric_threshold):
        deploy_model = mlengine_deploy_op(
        model_uri=train_model.outputs['job_dir'],
        project_id=project_id,
        model_id=model_id,
        version_id=version_id,
        runtime_version=RUNTIME_VERSION,
        python_version=PYTHON_VERSION,
        replace_existing_version=replace_existing_version)

    # Configure the pipeline to run using the service account defined
    # in the user-gcp-sa k8s secret
    if USE_KFP_SA == 'True':
        kfp.dsl.get_pipeline_conf().add_op_transformer(
              use_gcp_secret('user-gcp-sa'))
