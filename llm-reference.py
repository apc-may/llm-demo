# Databricks notebook source
!pip install xformers

# COMMAND ----------

import mlflow
import transformers

architecture = "databricks/dolly-v2-3b"

dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=dolly,
        artifact_path="dolly3b",
        input_example="Hello, Dolly!",
    )

#loaded_dolly = mlflow.transformers.load_model(
#    model_info.model_uri, 
#    max_new_tokens=250,
#)

#logged_model = 'runs:/c19c68f6cb8b48cb91366db215626c50/dolly3b'
#autolog_run = mlflow.last_active_run()
#model_uri = logged_model.format(autolog_run.info.run_id)
mlflow.register_model(model_info.model_uri, "dolly-v2-3b")

# COMMAND ----------



# COMMAND ----------


