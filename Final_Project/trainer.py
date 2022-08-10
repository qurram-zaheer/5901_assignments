
from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

# define the list of features in _FEATURE_KEYS variable
# 8 => CODE HERE # 
_FEATURE_KEYS = ['age','bmi','elective_surgery','height','pre_icu_los_days','weight','apache_2_diagnosis',
  'apache_3j_diagnosis','apache_post_operative','arf_apache','gcs_eyes_apache',
  'gcs_motor_apache','gcs_unable_apache','gcs_verbal_apache','heart_rate_apache','intubated_apache',
  'map_apache','resprate_apache','temp_apache','ventilated_apache','d1_diasbp_max','d1_diasbp_min',
  'd1_diasbp_noninvasive_max','d1_diasbp_noninvasive_min','d1_heartrate_max','d1_heartrate_min','d1_mbp_max',
  'd1_mbp_min','d1_mbp_noninvasive_max','d1_mbp_noninvasive_min','d1_resprate_max','d1_resprate_min','d1_spo2_max',
  'd1_spo2_min','d1_sysbp_max','d1_sysbp_min','d1_sysbp_noninvasive_max','d1_sysbp_noninvasive_min','d1_temp_max','d1_temp_min',
  'h1_diasbp_max','h1_diasbp_min','h1_diasbp_noninvasive_max','h1_diasbp_noninvasive_min','h1_heartrate_max','h1_heartrate_min',
  'h1_mbp_max','h1_mbp_min','h1_mbp_noninvasive_max','h1_mbp_noninvasive_min','h1_resprate_max','h1_resprate_min','h1_spo2_max','h1_spo2_min',
  'h1_sysbp_max','h1_sysbp_min','h1_sysbp_noninvasive_max','h1_sysbp_noninvasive_min','d1_glucose_max','d1_glucose_min','d1_potassium_max',
  'd1_potassium_min','apache_4a_hospital_death_prob','apache_4a_icu_death_prob','aids','cirrhosis','diabetes_mellitus','hepatic_failure',
  'immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis','cat-_african_american','cat-_asian','cat-_caucasian','cat-_hispanic',
  'cat-_native_american','cat-_other/unknown','cat-_f','cat-_m','cat-_ccu-cticu','cat-_csicu','cat-_cticu','cat-_cardiac_icu','cat-_micu',
  'cat-_med-surg_icu','cat-_neuro_icu','cat-_sicu','cat-_cardiovascular','cat-_gastrointestinal','cat-_genitourinary','cat-_gynecological',
  'cat-_hematological','cat-_metabolic','cat-_musculoskeletal/skin','cat-_neurological','cat-_respiratory','cat-_sepsis','cat-_trauma','cat-_haematologic',
  'cat-_neurologic','cat-_renal/genitourinary','cat-_undefined_diagnoses']

# define your target variable _LABEL_KEY
# 9 => CODE HERE # 
_LABEL_KEY = 'hospital_death'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

# Since we're not generating or creating a schema, we will instead create
# a feature spec.  Since there are a fairly small number of features this is
# manageable for this dataset.
_FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
           for feature in _FEATURE_KEYS
       },
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema).repeat()


def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  ####### MODEL ARCHITECTURE
  d = keras.layers.Dense(8, activation='relu')(d)
  d = keras.layers.Dense(8, activation='relu')(d)
  d = keras.layers.Dense(8, activation='relu')(d)

  outputs = keras.layers.Dense(1)(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(1e-2),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[keras.metrics.BinaryCrossentropy()])

  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # This schema is usually either an output of SchemaGen or a manually-curated
  # version provided by pipeline author. A schema can also derived from TFT
  # graph if a Transform component is used. In the case when either is missing,
  # `schema_from_feature_spec` could be used to generate schema from very simple
  # feature_spec, but the schema returned would be very primitive.
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
  ######## TRAIN THE DATASET
  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=_EVAL_BATCH_SIZE)

  model = _build_keras_model()
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')
  tf.saved_model.save(model, 'saved_model')
