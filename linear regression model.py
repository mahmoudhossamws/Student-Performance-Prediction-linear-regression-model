from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
from IPython.display import clear_output
from six.moves import urllib
import tensorflow_estimator


def clean_columns(df):
  df.columns = df.columns.str.replace(' ', '_').str.lower()
  return df

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use



print ("starting project")

# Load dataset.
dftrain = clean_columns( pd.read_csv('StudentsPerformance training.csv'))
dfeval = clean_columns(pd.read_csv('testing set.csv')) # testing data
y_train = dftrain.pop('math_score')
y_eval = dfeval.pop('math_score')

CATEGORICAL_COLUMNS = ['parental_level_of_education', 'lunch', 'test_preparation_course']
NUMERIC_COLUMNS = ['writing_score', 'reading_score']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train

predictions = list(linear_est.predict(eval_input_fn))
predicted_scores = np.array([p['predictions'][0] for p in predictions])

# Get actual scores
actual_scores = y_eval.values

clear_output()  # clears consoke output

results_df = pd.DataFrame({
    'Actual': actual_scores,
    'Predicted': predicted_scores,
    'Difference': actual_scores - predicted_scores,
    'accuracy': 1- abs(actual_scores - predicted_scores) / actual_scores,
    "writing_score": dfeval['writing_score'].values
})

results_df['accuracy'] = results_df['accuracy'].replace([-np.inf, np.inf], np.nan).clip(0, 1)

# 2. Create the plot (just 3 lines!)
plt.plot(results_df['accuracy'], 'b-')  # Plot the Accuracy column from DataFrame
plt.axhline(y=results_df['accuracy'].mean(), color='r', linestyle='--')  # Use mean from DataFrame
plt.title('Prediction Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Sample Index')
plt.show()



# 2. Now create the line plot
plt.figure(figsize=(10, 5))

# Sort by writing score for clean lines
results_df = results_df.sort_values('writing_score')

# Plot lines
plt.plot(results_df['writing_score'], results_df['Actual'], 'b-', label='Actual Math Score')
plt.plot(results_df['writing_score'], results_df['Predicted'], 'r--', label='Predicted Math Score')

# Add labels
plt.xlabel('Writing Score')
plt.ylabel('Math Score')
plt.title('Writing vs Math Scores')
plt.legend()
plt.grid(True)
plt.show()

print("\nFirst 10 Predictions vs Actual:")
print(results_df.head(10))