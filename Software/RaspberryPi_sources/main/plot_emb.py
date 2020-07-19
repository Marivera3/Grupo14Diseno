import cv2
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.svm import SVC
#from sklearn.preprocessing import LabelEncoder
'''
python3 plot_emb.py --embeddings output/embeddings_d2.pickle
'''

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
				help="path to serialized db of facial embeddings")
args = vars(ap.parse_args())


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

#le = LabelEncoder()
#labels = le.fit_transform(data["names"])

#model = SVC(kernel='linear', C=1E10)

X_embedded = TSNE(n_components=2).fit_transform(data['embeddings'])

#model.fit(X_embedded, labels)
#print(X_embedded)
plt.figure()
t = set(data['names'])
#colors = []
#col = {'b','r','y','g','m'}
for i,ta in enumerate(t):
	idx = np.where(np.array(data['names'])==ta)

	#print(colors.append(col[i]))
	#col.apppend([item for item in x for i in range(n)])
	plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=ta)
plt.title('Distribucion del dataset por SVM de kernel lineal')
plt.legend(bbox_to_anchor=(1, 1))
plt.grid()
plt.show()
'''
#plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, s=50, cmap='autumn')
#plot_svc_decision_function(model);

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
X0, X1 = X_embedded[:, 0], X_embedded[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c={'b','r','y','g','m'}, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
'''
