# Go to models/classification and run the file random_forest_decision_tree.py
# to see the explainer dashboard (might take a few minutes to start)
# OR
# check data/eda.py for streamlit dashboard


# some dashboard ideas here

# Explainer dashboard takes forever to calculate shap values
# Actually, by trying manually calculate SHAP I hang out the machine :)
# checking streamlit again

# from explainerdashboard import ClassifierExplainer, ExplainerDashboard
# from explainerdashboard.datasets import feature_descriptions

"""
explainer = ClassifierExplainer(model_name, X_test, y_test, 
                               descriptions=feature_descriptions,
                               labels=['Existing Customer', 'Attrited Customer'])
# Warning: calculating shap interaction values can be slow! Pass shap_interaction=False to remove interactions tab.
ExplainerDashboard(explainer, title="babushka churn prediction model dashboard", shap_interaction=False, simple=False).run() # simple=True, shap_interaction=False
"""

