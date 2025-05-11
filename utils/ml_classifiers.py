from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=100),
    # 'mlp': MLPClassifier(max_iter=1000),
    'svm': SVC(probability=True, kernel="rbf", gamma="scale"),
    # 'lr': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
}

# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
ensemble_clf = VotingClassifier(
    estimators=[
        ('mlp', CLASSIFIERS.get('mlp')),
        ('svm', CLASSIFIERS.get('svm')),
        ('rf', CLASSIFIERS.get('rf')),
        ('lr', CLASSIFIERS.get('lr')),
    ],
    voting='soft'
)
