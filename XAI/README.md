## Little info about XAI (Explainable AI)
### State of the art XAI methods are Shapley Values, LIME and ELI5. Logic behind all of them are model agnostic approaches so one can easily apply one of them with any trained model.
### The easiest, and the most primitive, way to understands your ML model results is to apply a DT algorithm and check the rules (aka check node split values). But even a simple DT Model might be huge tree that one can not easily observe the rules. At such point DecisionTreeRules.py just dumps the human-readible rules. 
