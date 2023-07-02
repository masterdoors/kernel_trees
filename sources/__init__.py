from .regularization import RefinedForestClassifier
from .regularization import RefinedForestRegressor
from .GPU import GPUForestClassifier
from .GPU import GPUForestRegressor
from .CO2_forest import CO2ForestRegressor
from .CO2_forest import CO2ForestClassifier

__all__ = ["RefinedForestClassifier",
            "RefinedForestRegressor", 
           "GPUForestClassifier",
            "GPUForestRegressor",
            "CO2ForestRegressor",
             "CO2ForestClassifier"]

