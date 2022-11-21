import tensorflow as tf
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import statsmodels.api as sm

from tensorflow_probability import edward2 as ed
from sklearn.datasets import load_breast_cancer
from pandas.plotting import scatter_matrix
from scipy import sparse, stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams.update({'font.sans-serif': 'Helvetica',
                            'axes.labelsize': 10,
                            'xtick.labelsize': 6,
                            'ytick.labelsize': 6,
                            'axes.titlesize': 10})


## TODO: remove later
def load_data():
    data = load_breast_cancer()
    # work with the first 10 features
    num_fea = 10
    df = pd.DataFrame(data["data"][:, :num_fea], columns=data["feature_names"][:num_fea])
    dfy = data["target"]
    return df, dfy


## TODO: make a scatter plot of all pairs of the causes, exclude highly correlated causes
def plot_scatter(df):
    sns.pairplot(df, size=1.5)
    plt.show()


def drop_high_correlated(df):
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    dfX = df.drop(columns=to_drop)
    print('df1\n', dfX)
    return dfX


def split_train_val(X):
    num_datapoints, data_dim = X.shape
    holdout_portion = 0.2
    n_holdout = int(holdout_portion * num_datapoints * data_dim)

    holdout_row = np.random.randint(num_datapoints, size=n_holdout)
    holdout_col = np.random.randint(data_dim, size=n_holdout)
    holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), (holdout_row, holdout_col)), shape=X.shape)).toarray()

    holdout_subjects = np.unique(holdout_row)
    holdout_mask = np.minimum(1, holdout_mask)

    x_train = np.multiply(1 - holdout_mask, X)
    x_vad = np.multiply(holdout_mask, X)
    return x_train, x_vad, holdout_subjects, holdout_mask, holdout_row


def process_data():
    df, dfy = load_data()
    plot_scatter(df)
    dfX = drop_high_correlated(df)
    # standardize the data for PPCA
    X = np.array((dfX - dfX.mean()) / dfX.std())
    num_datapoints, data_dim = X.shape
    x_train, x_vad, holdout_subjects, holdout_mask, holdout_row = split_train_val(X)
    return X, dfy, x_train, x_vad, holdout_subjects, holdout_mask, holdout_row, num_datapoints, data_dim


# a probabilistic PCA model
# we allow both linear and quadratic model
# for linear model x_n has mean z_n * W
# for quadratic model x_n has mean b + z_n * W + (z_n**2) * W_2
# quadratic model needs to change the checking step accordingly
def ppca_model(data_dim, latent_dim, num_datapoints, stddv_datapoints, mask, form="linear"):
    w = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                  scale=tf.ones([latent_dim, data_dim]),
                  name="w")  # parameter
    z = ed.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                  scale=tf.ones([num_datapoints, latent_dim]),
                  name="z")  # local latent variable / substitute confounder
    if form == "linear":
        # mean z_n * W
        x = ed.Normal(loc=tf.multiply(tf.matmul(z, w), mask),
                      scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                      name="x")  # (modeled) data
    elif form == "quadratic":
        b = ed.Normal(loc=tf.zeros([1, data_dim]),
                      scale=tf.ones([1, data_dim]),
                      name="b")  # intercept
        w2 = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                       scale=tf.ones([latent_dim, data_dim]),
                       name="w2")  # quadratic parameter
        # mean b + z_n * W + (z_n**2) * W_2
        x = ed.Normal(loc=tf.multiply(b + tf.matmul(z, w) + tf.matmul(tf.square(z), w2), mask),
                      scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                      name="x")  # (modeled) data
    return x, (w, z)


log_joint = ed.make_log_joint_fn(ppca_model)


def target(b, w, w2, z, x_train, data_dim, latent_dim, num_datapoints, stddv_datapoints, holdout_mask):
    return log_joint(data_dim=data_dim,
                     latent_dim=latent_dim,
                     num_datapoints=num_datapoints,
                     stddv_datapoints=stddv_datapoints,
                     mask=1 - holdout_mask,
                     w=w, z=z, w2=w2, b=b, x=x_train)


# log_joint = ed.make_log_joint_fn(ppca_model)
# variational inference for probabilistic PCA
def variational_model(qb_mean, qb_stddv, qw_mean, qw_stddv, qw2_mean, qw2_stddv, qz_mean, qz_stddv):
    qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")
    qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
    qw2 = ed.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
    qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
    return qb, qw, qw2, qz


log_q = ed.make_log_joint_fn(variational_model)


def target_q(qb, qw, qw2, qz, qb_mean, qb_stddv, qw_mean, qw_stddv, qw2_mean, qw2_stddv, qz_mean, qz_stddv):
    return log_q(qb_mean=qb_mean, qb_stddv=qb_stddv,
                 qw_mean=qw_mean, qw_stddv=qw_stddv,
                 qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                 qz_mean=qz_mean, qz_stddv=qz_stddv,
                 qw=qw, qz=qz, qw2=qw2, qb=qb)


def get_inferred(x_train, latent_dim, num_datapoints, data_dim, stddv_datapoints, holdout_mask):
    qb_mean = tf.Variable(np.ones([1, data_dim]), dtype=tf.float32)
    qw_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
    qw2_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
    qz_mean = tf.Variable(np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
    qb_stddv = tf.nn.softplus(tf.Variable(0 * np.ones([1, data_dim]), dtype=tf.float32))
    qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
    qw2_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
    qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([num_datapoints, latent_dim]), dtype=tf.float32))

    qb, qw, qw2, qz = variational_model(qb_mean=qb_mean, qb_stddv=qb_stddv,
                                        qw_mean=qw_mean, qw_stddv=qw_stddv,
                                        qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                                        qz_mean=qz_mean, qz_stddv=qz_stddv)

    energy = target(qb, qw, qw2, qz, x_train, data_dim, latent_dim, num_datapoints, stddv_datapoints, holdout_mask)
    entropy = -target_q(qb, qw, qw2, qz, qb_mean, qb_stddv, qw_mean, qw_stddv, qw2_mean, qw2_stddv, qz_mean, qz_stddv)
    # evidence lower bound log(P(X,Z) / Q(Z)) = log(P(X,Z)) - log(Q(Z))
    # maximizing the ELBO (min -elbo) would simultaneously allow us to obtain an accurate generative model
    elbo = energy + entropy

    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    train = optimizer.minimize(-elbo)

    init = tf.global_variables_initializer()

    t = []

    num_epochs = 500

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_epochs):
            sess.run(train)
            if i % 5 == 0:
                t.append(sess.run([elbo]))

            b_mean_inferred = sess.run(qb_mean)
            b_stddv_inferred = sess.run(qb_stddv)
            w_mean_inferred = sess.run(qw_mean)
            w_stddv_inferred = sess.run(qw_stddv)
            w2_mean_inferred = sess.run(qw2_mean)
            w2_stddv_inferred = sess.run(qw2_stddv)
            z_mean_inferred = sess.run(qz_mean)
            z_stddv_inferred = sess.run(qz_stddv)

    print("Inferred axes:")
    print(w_mean_inferred)
    print("Standard Deviation:")
    print(w_stddv_inferred)

    plt.plot(range(1, num_epochs, 5), t)
    plt.show()
    inferred_dic = {'b_mean_inferred': b_mean_inferred,
                    'b_stddv_inferred': b_stddv_inferred,
                    'w_mean_inferred': w_mean_inferred,
                    'w_stddv_inferred': w_stddv_inferred,
                    'w2_mean_inferred': w2_mean_inferred,
                    'w2_stddv_inferred': w2_stddv_inferred,
                    'z_mean_inferred': z_mean_inferred,
                    'z_stddv_inferred': z_stddv_inferred}
    return inferred_dic


def replace_latents(b, w, w2, z):

    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
        """Replaces the priors with actual values to generate samples from."""
        name = rv_kwargs.pop("name")
        if name == "b":
            rv_kwargs["value"] = b
        elif name == "w":
            rv_kwargs["value"] = w
        elif name == "w":
            rv_kwargs["value"] = w2
        elif name == "z":
            rv_kwargs["value"] = z
        return rv_constructor(*rv_args, **rv_kwargs)

    return interceptor


def replicated_data(inferred_dic, x_train, latent_dim, num_datapoints, data_dim, stddv_datapoints, holdout_mask):
    n_rep = 100  # number of replicated datasets we generate
    holdout_gen = np.zeros((n_rep, *x_train.shape))

    for i in range(n_rep):
        b_sample = npr.normal(inferred_dic['b_mean_inferred'], inferred_dic['b_stddv_inferred'])
        w_sample = npr.normal(inferred_dic['w_mean_inferred'], inferred_dic['w_stddv_inferred'])
        w2_sample = npr.normal(inferred_dic['w2_mean_inferred'], inferred_dic['w2_stddv_inferred'])
        z_sample = npr.normal(inferred_dic['z_mean_inferred'], inferred_dic['z_stddv_inferred'])

        with ed.interception(replace_latents(b_sample, w_sample, w2_sample, z_sample)):
            generate = ppca_model(data_dim=data_dim,
                                  latent_dim=latent_dim,
                                  num_datapoints=num_datapoints,
                                  stddv_datapoints=stddv_datapoints,
                                  mask=np.ones(x_train.shape))

        with tf.Session() as sess:
            x_generated, _ = sess.run(generate)

        # look only at the heldout entries
        holdout_gen[i] = np.multiply(x_generated, holdout_mask)
    return holdout_gen


def test_stat(inferred_dic, x_train, x_vad, latent_dim, num_datapoints, data_dim, stddv_datapoints, holdout_mask):
    holdout_gen = replicated_data(inferred_dic,
                                  x_train,
                                  latent_dim,
                                  num_datapoints,
                                  data_dim,
                                  stddv_datapoints,
                                  holdout_mask)
    n_eval = 100  # we draw samples from the inferred Z and W
    obs_ll = []
    rep_ll = []
    for j in range(n_eval):
        w_sample = npr.normal(inferred_dic['w_mean_inferred'], inferred_dic['w_stddv_inferred'])
        z_sample = npr.normal(inferred_dic['z_mean_inferred'], inferred_dic['z_stddv_inferred'])

        holdoutmean_sample = np.multiply(z_sample.dot(w_sample), holdout_mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, stddv_datapoints).logpdf(x_vad), axis=1))

        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, stddv_datapoints).logpdf(holdout_gen), axis=2))

    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
    return obs_ll_per_zi, rep_ll_per_zi


def p_value(inferred_dic, x_train, x_vad, latent_dim, num_datapoints, data_dim, stddv_datapoints, holdout_mask, holdout_row):
    obs_ll_per_zi, rep_ll_per_zi = test_stat(inferred_dic,
                                             x_train,
                                             x_vad,
                                             latent_dim,
                                             num_datapoints,
                                             data_dim,
                                             stddv_datapoints,
                                             holdout_mask)
    pvals = np.array([np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(holdout_row)
    overall_pval = np.mean(pvals[holdout_subjects])
    print('overall_pval', overall_pval)
    return overall_pval


def main():
    X, dfy, x_train, x_vad, holdout_subjects, holdout_mask, holdout_row, num_datapoints, data_dim = process_data()
    # fit a probabilistic PCA model to x_train
    latent_dim = 2
    stddv_datapoints = 0.1
    model = ppca_model(data_dim=data_dim,
                       latent_dim=latent_dim,
                       num_datapoints=num_datapoints,
                       stddv_datapoints=stddv_datapoints,
                       mask=1 - holdout_mask)
    inferred_dic = get_inferred(x_train,
                                latent_dim,
                                num_datapoints,
                                data_dim,
                                stddv_datapoints,
                                holdout_mask)
    overall_pval = p_value(inferred_dic,
                           x_train,
                           x_vad,
                           latent_dim,
                           num_datapoints,
                           data_dim,
                           stddv_datapoints,
                           holdout_mask,
                           holdout_row)
    if overall_pval > 0.01:
        # approximate the (random variable) substitute confounders with their inferred mean.
        Z_hat = inferred_dic['z_mean_inferred']
        # augment the regressors to be both the assigned causes X and the substitute confounder Z
        X_aug = np.column_stack([X, Z_hat])
        # holdout some data from prediction later
        X_train, X_test, y_train, y_test = train_test_split(X_aug, dfy, test_size=0.2, random_state=0)
        dcfX_train = sm.add_constant(X_train)
        dcflogit_model = sm.Logit(y_train, dcfX_train)
        dcfresult = dcflogit_model.fit_regularized(maxiter=5000)
        print(dcfresult.summary())
        # make predictions with the causal model
        dcfX_test = X_test
        dcfy_predprob = dcfresult.predict(sm.add_constant(dcfX_test))
        dcfy_pred = (dcfy_predprob > 0.5)
        print(classification_report(y_test, dcfy_pred))
    return X_aug, dfy


if __name__ == "__main__":
    main()
