import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from IPython import display

# display.set_matplotlib_formats('svg')

class Iris(nn.Module):
    """
    Iris ANN Model
    """
    def __init__(self) -> None:
      super().__init__()
      self.input = nn.Linear(4,128)
      self.fc1 = nn.Linear(128,128)
      self.fc2 = nn.Linear(128,128)
      self.output = nn.Linear(128,3)

    def forward(self,x) -> nn.Linear:
      x = F.relu(self.input(x))
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return self.output(x)
    
    def _load_model(self) -> None:
      """
      Loads the trained model
      """
      self.load_state_dict(torch.load("trainedModel.pt"))

    def predict(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
      """
      It will predict the flower type from given data
      Arguments:
          sepal_length: float
          sepal_width: float
          petal_length: float
          petal_width: float
      Returns:
          Predicted flower type
      """
      self._load_model()
      d = {'sepal_length': [sepal_length] ,'sepal_width':[sepal_width],'petal_length':[petal_length],'petal_width':[petal_width] }
      df = pd.DataFrame(data=d)
      tensor_data = torch.tensor( df[df.columns[0:4]].values ).float()
      predictions = self(tensor_data)
      iris_flowers = ["Setosa","Versicolor","Virginica"]
      predicted_number = torch.argmax(predictions,axis=1)
      return iris_flowers[predicted_number]