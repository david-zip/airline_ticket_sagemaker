# import relevant modules
import os
import boto3
import sagemaker
from datetime import datetime

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString
)

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor
)

from sagemaker.workflow.steps import (
    ProcessingStep,
    TuningStep,
    TrainingStep,
    HyperparameterTuner,
    Model
)

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from sagemaker.workflow.condition_step import (
    ConditionStep,
)

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoostPredictor
from sagemaker.parameter import IntegerParameter
from sagemaker.sklearn.processing import SKLearnProcessor

BASE_DIR = os.path.abspath('')

nw = datetime.now()

def get_pipeline(
    base_job_name = f'AirlineTicket',
    default_bucket = 'sagemaker-ticket-price',
    pipeline_name = f'{nw.year}-{nw.month}-{nw.day}({nw.hour}-{nw.minute})-Airline-Ticket-Price',
    region = 'eu-west-1'
):
    # initialise sagemaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client('sagemaker')
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket
    )

    pipeline_session = sagemaker.workflow.pipeline_context.PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket
    )

    role = sagemaker.session.get_execution_role(sagemaker_session) 

    # initialise pipeline parameters
    processing_instance_count = 1

    processing_instance_type = r'ml.m5.xlarge'

    model_approval_status = 'PendingManualApproval'

    input_data = 's3://sagemaker-ticket-price/data/airline_ticket.csv'

    # preprocess data
    sklearn_processor = SKLearnProcessor(
        framework_version='0.23-1',
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f'{base_job_name}-sklearn-ticket-price-preprocessing',
        sagemaker_session=pipeline_session,
        role=role
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(
                output_name='train',
                source='/opt/ml/processing/train'
            ),
            ProcessingOutput(
                output_name='test',
                source='/opt/ml/processing/test'
            ),
            ProcessingOutput(
                output_name='validation',
                source='/opt/ml/processing/validation'
            )
        ],
        code=os.path.join(BASE_DIR, 'preprocess.py'),
        arguments=['--input-data', input_data]
    )

    step_process = ProcessingStep(
        name=f'Preprocessing{base_job_name}',
        step_args=step_args
    )

    # initialise train xgboost model
    model_path = f's3://{sagemaker_session.default_bucket()}/{base_job_name}/AirlineTicketTrain'

    # initialise xgboost training algorithm
    image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.0-1',
        py_version='py3',
        instance_type=processing_instance_type
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        output_path=model_path,
        base_job_name=f'{base_job_name}/airline-ticket-training',
        sagemaker_session=pipeline_session,
        role=role
    )

    xgb_train.set_hyperparameters(
        eval_metric='rmse',
        objective="reg:squarederror",
        num_round=50,
        min_child_weight=6,
        subsample=0.5,
        silent=0
    )

    # initialise hyperparameter tuner
    objective_metric_name = 'validation:rmse'

    hyperparameter_ranges = {
        'max_depth': IntegerParameter(min_value=6, max_value=9, scaling_type='Linear')
    }

    tuner_log = HyperparameterTuner(
        estimator=xgb_train,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy='random'
    )

    # train xgboost model
    step_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    step_tune = TuningStep(
        name="HPOTuningAirlinePrice",
        step_args=step_args
    )
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_name}/scriptAirlinePriceEval",
        sagemaker_session=pipeline_session,
        role=role
    )

    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_tune.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation"
            )
        ],
        code=os.path.join(BASE_DIR, "evaluate.py")
    )

    evaluation_report = PropertyFile(
        name="HousePriceEvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name="EvaluateAirlinePriceModel",
        step_args=step_args,
        property_files=[evaluation_report]
    )

    # create sagemaker model from training job
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    model_prefix = f'{base_job_name}/AirlineTicketTrain'

    model = Model(
        image_uri=image_uri,
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=default_bucket,
            prefix=model_prefix
        ),
        predictor_cls=XGBoostPredictor,
        sagemamer_session=sagemaker_session,
        role=role
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )

    step_register = ModelStep(
        name='RegisterAirlinePriceModel',
        step_args=step_args
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=10.0
    )
    step_cond = ConditionStep(
        name="CheckMSEAirlinePriceEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[]
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[
            step_process,
            step_tune,
            step_eval,
            step_cond
        ],
        sagemaker_session=pipeline_session
    )

    return pipeline
