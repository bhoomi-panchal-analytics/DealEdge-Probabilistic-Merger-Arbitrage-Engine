from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ProbabilityModel:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, df, features, target='status'):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model.fit(X_train, y_train)
        return self.model

    def predict_prob(self, df, features):
        return self.model.predict_proba(df[features])[:, 1]
