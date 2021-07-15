"""Helper components."""

from typing import NamedTuple

def prepoc_split_dataset(root_path: str
)-> NamedTuple('Outputs', [('training_file_path', str), ('validation_file_path', str),
                            ('testing_file_path', str)]):

  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn import preprocessing
  import os

  os.system('curl -o iris.txt  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    
  os.system('gsutil cp  iris.txt  {}'.format(root_path))
    
    
  columns = ['sepal_length','sepal_width','petal_length','petal_width','class']

  dfIris = pd.read_csv('{}/iris.txt'.format(root_path), names=columns)
  le = preprocessing.LabelEncoder()
  dfIris['classEncoder'] = le.fit_transform(dfIris['class'])
    
  X_train, X_val_test, y_train, y_val_test = train_test_split(dfIris.drop(['classEncoder','class'], axis=1), dfIris['classEncoder'], test_size=0.2)

  X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test, test_size=0.5)
    
  df_train = pd.concat([X_train,y_train],axis=1)
  df_val = pd.concat([X_val,y_val],axis=1)
  df_test = pd.concat([X_test,y_test],axis=1)
  
  df_train.to_csv('train.txt', index=False)
  df_val.to_csv('validation.txt', index=False)
  df_test.to_csv('test.txt' , index=False)

  os.system('gsutil cp  train.txt  {}'.format(root_path))
  os.system('gsutil cp  validation.txt  {}'.format(root_path))
  os.system('gsutil cp  test.txt  {}'.format(root_path))
  
  training_file_path = '{}/train.txt'.format(root_path)
  validation_file_path = '{}/validation.txt'.format(root_path)
  testing_file_path = '{}/test.txt'.format(root_path)
    
  return (training_file_path,validation_file_path,testing_file_path)

def retrieve_best_run(
    project_id: str, job_id: str
) -> NamedTuple('Outputs', [('metric_value', float), ('alpha', float),
                            ('max_iter', int)]):
  """Retrieves the parameters of the best Hypertune run."""

  from googleapiclient import discovery
  from googleapiclient import errors

  ml = discovery.build('ml', 'v1')

  job_name = 'projects/{}/jobs/{}'.format(project_id, job_id)
  request = ml.projects().jobs().get(name=job_name)

  try:
    response = request.execute()
  except errors.HttpError as err:
    print(err)
  except:
    print('Unexpected error')

  print(response)

  best_trial = response['trainingOutput']['trials'][0]

  metric_value = best_trial['finalMetric']['objectiveValue']
  alpha = float(best_trial['hyperparameters']['alpha'])
  max_iter = int(best_trial['hyperparameters']['max_iter'])

  return (metric_value, alpha, max_iter)


def evaluate_model(
    testing_file_path: str, model_path: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float),
                            ('mlpipeline_metrics', 'Metrics')]):
  """Evaluates a trained sklearn model."""
  #import joblib
  import pickle
  import json
  import pandas as pd
  import subprocess
  import sys

  from sklearn.metrics import accuracy_score, recall_score
  

  df_test = pd.read_csv(testing_file_path)

  X_test = df_test.drop('classEncoder', axis=1)
  y_test = df_test['classEncoder']

  # Copy the model from GCS
  model_filename = 'model.pkl'
  gcs_model_filepath = '{}/{}'.format(model_path, model_filename)
  print(gcs_model_filepath)
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, model_filename],
                        stderr=sys.stdout)

  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

  y_hat = model.predict(X_test)

  if metric_name == 'accuracy':
    metric_value = accuracy_score(y_test, y_hat)
  elif metric_name == 'recall':
    metric_value = recall_score(y_test, y_hat)
  else:
    metric_name = 'N/A'
    metric_value = 0

  # Export the metric
  metrics = {
      'metrics': [{
          'name': metric_name,
          'numberValue': float(metric_value)
      }]
  }

  return (metric_name, metric_value, json.dumps(metrics))

def custom_deploy(model_uri: str, project_id: str,model_id: str,version_id: str):
  import subprocess
  import sys
  
  model_filename = 'model.pkl'
  gcs_model_filepath = '{}/{}'.format(model_uri, model_filename)
  print(gcs_model_filepath)
  subprocess.check_call(['git', 'pull', 'https://github.com/alantellecom/custom_gcp_kubeflow.git'],
                        stderr=sys.stdout)
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, 'deployApi/{}'.format(model_filename)],
                        stderr=sys.stdout)
  subprocess.check_call(['gcloud', 'builds', 'submit', '--config', 'deployApi/cloudbuild.yaml',
            '--substitutions', '_PROJECT_ID={},_API_NAME={},_VERSION={}'.format(project_id,model_id,version_id)] ,stderr=sys.stdout)

                        
  subprocess.check_call(['gsutil', 'run', 'deploy', model_id, '--image gcr.io/{}/{}:{}'.format(project_id,model_id,version_id)],
                        stderr=sys.stdout)
