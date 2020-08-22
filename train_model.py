# import the necessery packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#import argparse
import pickle
import h5py
import config.fruits_ml_web_app as config

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--db", required = True,
#                help = "path HDF5 database")
#ap.add_argument("-m", "--model", required = True,
#                help = "path to output model")
#ap.add_argument("-j", "--jobs", type = int, default = -1,
#                help = "# of jobs to run when tuning hyperparameters")
#args = vars(ap.parse_args())

# open the HDF5 dataset for reading than determine the index of
# the traininig and testing split, provided that this data was
# already shuffled *prior* to writing to disk
#db = h5py.File(args["db"], "r")
db = h5py.File(config.FEATURES_HDF5, "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters that we want to tune than start a 
# grid search where we evaluate our model for each value o C
print("[INFO] tuning hyperparameters...")
# GreadSearch (in sklearn) automatically iterates over these hyperparameters using cross validation
# GridSearchCV will automatically search for the best hyper-parameter from the given params by using and therefore 
# after training evaluating cost
params = {"C" : [0.0001, 0.001, 0.01, 0.1, 1.0]} # free parameters in dictionary (or list of dictionaries if multiple params)
#model = GridSearchCV(LogisticRegression(), params, cv = 3,
#                     n_jobs = args["jobs"])
## GridSearchCV(estimator, param_grid, scoring, cv, n_jobs)
model = GridSearchCV(LogisticRegression(max_iter = 1000), params, cv = 3, # cv means cross validation
                     n_jobs = config.N_JOBS)
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_)) # this will print the best parameters it has found after a grid search

# generate a classification report for the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
                            target_names = db["label_names"]))

# compare the raw accuracy with extra precision
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

# serialize the model
print("[INFO] saving model...")
#f = open(args["model"], "wb")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()






























