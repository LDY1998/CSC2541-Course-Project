import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import random
from scipy import sparse, stats
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

rand_seed = 123
INPUT_PATH = "../expert_data/Trajectories-10_samples-10000_masked-5_confounded.pkl"
expert_path = "../expert_data/Hopper-v2.pkl"
confounded_path = "../expert_data/Trajectories-10_samples-10000_confounded.pkl"


def set_rand_seed():
    # set random seed so everyone gets the same number
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)


def without_keys(dic, keys):
    return {k: dic[k] for k in dic if k not in keys}


def load_data(expert_path, drop_dims):
    # load masked data with observed confounder
    with open(expert_path, 'rb') as handle:
        dic = pickle.load(handle)
        df = pd.DataFrame(dic['observations'],
                          columns=['observation_' + str(i + 1) for i in range(dic['observations'].shape[1])])
        df_masked = df.drop(df.columns[drop_dims], axis=1)
        X, dict_distribution = process_data(df_masked)
        # select all columns except 'observations'
        dic_rest = without_keys(dic, {"observations"})
        return X, dict_distribution, dic_rest


def pickle_data(dic, dim):
    # save dictionary to pickle file
    OUTPUT_PATH = f"../expert_data/Trajectories-10_samples-10000_masked-5_confounded_inferred-{dim}.pkl"
    with open(OUTPUT_PATH, 'wb') as file:
        pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)


def plot_scatter(df):
    # make a scatter plot of all pairs of the causes, exclude highly correlated causes
    sns.pairplot(df, size=1.5)
    plt.show()


def drop_high_correlated(df):
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    dfX = df.drop(columns=to_drop)
    return dfX


def process_data(df):
    # plot_scatter(df)
    dfX = drop_high_correlated(df)
    dict_distribution = {
        'mean': dfX.mean().to_numpy(),
        'std': dfX.std().to_numpy()
    }
    X = np.array((dfX - dfX.mean()) / dfX.std())
    return X, dict_distribution


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


class Deconfounder:
    def __init__(
            self,
            X,
            latent_dim
    ):
        self.X = X
        self.latent_dim = latent_dim
        self.num_datapoints, self.data_dim = self.X.shape
        self.x_train, self.x_vad, self.holdout_subjects, self.holdout_mask, self.holdout_row = split_train_val(self.X)
        self.stddv_datapoints = 0.1
        self.qb_mean = tf.Variable(np.ones([1, self.data_dim]), dtype=tf.float32)
        self.qw_mean = tf.Variable(np.ones([self.latent_dim, self.data_dim]), dtype=tf.float32)
        self.qw2_mean = tf.Variable(np.ones([self.latent_dim, self.data_dim]), dtype=tf.float32)
        self.qz_mean = tf.Variable(np.ones([self.num_datapoints, self.latent_dim]), dtype=tf.float32)
        self.qb_stddv = tf.nn.softplus(tf.Variable(0 * np.ones([1, self.data_dim]), dtype=tf.float32))
        self.qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.latent_dim, self.data_dim]), dtype=tf.float32))
        self.qw2_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.latent_dim, self.data_dim]), dtype=tf.float32))
        self.qz_stddv = tf.nn.softplus(
            tf.Variable(-4 * np.ones([self.num_datapoints, self.latent_dim]), dtype=tf.float32))

    # a probabilistic PCA model
    # we allow both linear and quadratic model
    # for linear model x_n has mean z_n * W
    # for quadratic model x_n has mean b + z_n * W + (z_n**2) * W_2
    # quadratic model needs to change the checking step accordingly
    def ppca_model(self, data_dim, latent_dim, num_datapoints, stddv_datapoints, mask, form="linear"):
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

    def target(self, b, w, w2, z):
        return self.log_joint(data_dim=self.data_dim,
                              latent_dim=self.latent_dim,
                              num_datapoints=self.num_datapoints,
                              stddv_datapoints=self.stddv_datapoints,
                              mask=1 - self.holdout_mask,
                              w=w, z=z, w2=w2, b=b, x=self.x_train)

    # log_joint = ed.make_log_joint_fn(ppca_model)
    # variational inference for probabilistic PCA
    def variational_model(self, qb_mean, qb_stddv, qw_mean, qw_stddv, qw2_mean, qw2_stddv, qz_mean, qz_stddv):
        qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")
        qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
        qw2 = ed.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
        qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        return qb, qw, qw2, qz

    log_q = ed.make_log_joint_fn(variational_model)

    def target_q(self, qb, qw, qw2, qz):
        return self.log_q(qb_mean=self.qb_mean, qb_stddv=self.qb_stddv,
                          qw_mean=self.qw_mean, qw_stddv=self.qw_stddv,
                          qw2_mean=self.qw2_mean, qw2_stddv=self.qw2_stddv,
                          qz_mean=self.qz_mean, qz_stddv=self.qz_stddv,
                          qw=qw, qz=qz, qw2=qw2, qb=qb)

    def get_inferred(self):
        qb, qw, qw2, qz = self.variational_model(qb_mean=self.qb_mean, qb_stddv=self.qb_stddv,
                                                 qw_mean=self.qw_mean, qw_stddv=self.qw_stddv,
                                                 qw2_mean=self.qw2_mean, qw2_stddv=self.qw2_stddv,
                                                 qz_mean=self.qz_mean, qz_stddv=self.qz_stddv)

        energy = self.target(qb, qw, qw2, qz)
        entropy = -self.target_q(qb, qw, qw2, qz)
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

                b_mean_inferred = sess.run(self.qb_mean)
                b_stddv_inferred = sess.run(self.qb_stddv)
                w_mean_inferred = sess.run(self.qw_mean)
                w_stddv_inferred = sess.run(self.qw_stddv)
                w2_mean_inferred = sess.run(self.qw2_mean)
                w2_stddv_inferred = sess.run(self.qw2_stddv)
                z_mean_inferred = sess.run(self.qz_mean)
                z_stddv_inferred = sess.run(self.qz_stddv)

        # print("Inferred axes:", w_mean_inferred)
        # print("Standard Deviation:", w_stddv_inferred)

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

    def replicated_data(self, inferred_dic):
        n_rep = 100  # number of replicated datasets we generate
        holdout_gen = np.zeros((n_rep, *self.x_train.shape))

        for i in range(n_rep):
            b_sample = npr.normal(inferred_dic['b_mean_inferred'], inferred_dic['b_stddv_inferred'])
            w_sample = npr.normal(inferred_dic['w_mean_inferred'], inferred_dic['w_stddv_inferred'])
            w2_sample = npr.normal(inferred_dic['w2_mean_inferred'], inferred_dic['w2_stddv_inferred'])
            z_sample = npr.normal(inferred_dic['z_mean_inferred'], inferred_dic['z_stddv_inferred'])

            with ed.interception(replace_latents(b_sample, w_sample, w2_sample, z_sample)):
                generate = self.ppca_model(data_dim=self.data_dim,
                                           latent_dim=self.latent_dim,
                                           num_datapoints=self.num_datapoints,
                                           stddv_datapoints=self.stddv_datapoints,
                                           mask=np.ones(self.x_train.shape))

            with tf.Session() as sess:
                x_generated, _ = sess.run(generate)

            # look only at the heldout entries
            holdout_gen[i] = np.multiply(x_generated, self.holdout_mask)
        return holdout_gen

    def test_stat(self, inferred_dic):
        holdout_gen = self.replicated_data(inferred_dic)
        n_eval = 100  # we draw samples from the inferred Z and W
        obs_ll = []
        rep_ll = []
        for j in range(n_eval):
            w_sample = npr.normal(inferred_dic['w_mean_inferred'], inferred_dic['w_stddv_inferred'])
            z_sample = npr.normal(inferred_dic['z_mean_inferred'], inferred_dic['z_stddv_inferred'])

            holdoutmean_sample = np.multiply(z_sample.dot(w_sample), self.holdout_mask)
            obs_ll.append(np.mean(stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(self.x_vad), axis=1))
            rep_ll.append(np.mean(stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(holdout_gen), axis=2))

        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
        return obs_ll_per_zi, rep_ll_per_zi

    def p_value(self, inferred_dic):
        obs_ll_per_zi, rep_ll_per_zi = self.test_stat(inferred_dic)
        pvals = np.array([np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i]) for i in range(self.num_datapoints)])
        holdout_subjects = np.unique(self.holdout_row)
        overall_pval = np.mean(pvals[holdout_subjects])
        print('overall_pval', overall_pval)
        return overall_pval

    def substitute(self):
        inferred_dic = self.get_inferred()
        overall_pval = self.p_value(inferred_dic)
        if overall_pval > 0.01:
            # approximate the (random variable) substitute confounders with their inferred mean.
            Z_hat = inferred_dic['z_mean_inferred']
            return Z_hat
        else:
            raise ValueError("p-value is too small!")


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


def factor_model(confounded, drop_dims, latent_dim):
    set_rand_seed()
    save_path = "../generated_data/Trajectories-10_samples-10000"
    if confounded:
        save_path += "_confounded"

    if len(drop_dims) != 0:  # masked
        save_path += f"_masked-{drop_dims}"

    save_file = Path(save_path)
    if not save_file.is_file():
        data = {}
        input_path = confounded_path if confounded else expert_path
        X, dict_distribution, _ = load_data(input_path, drop_dims)
        if latent_dim == -1:  # not use factor model
            save_path += ".pkl"
            data = {'npz_dic': {**dict_distribution, 'zs': None},
                    'regr': None}
        else:
            save_path += f"_inferred-{latent_dim}.pkl"
            deconfounder = Deconfounder(X, latent_dim)
            z = deconfounder.substitute()
            regr = MultiOutputRegressor(MLPRegressor(random_state=1, max_iter=500))
            regr.fit(X, z)
            data = {'npz_dic': {**dict_distribution, 'zs': z},
                    'regr': regr}
        with open(save_path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    factor_model(True, [10], 3) #  confounded + mask-[10] + inferred-3
    factor_model(True, [10], -1)  # confounded + mask-[10] + uninferred
    path = "../generated_data/Trajectories-10_samples-10000_confounded_masked-[10]_inferred-3.pkl"
    with open(path, 'rb') as handle:
        load = pickle.load(handle)
        npz_dic = load['npz_dic']
        regr = load['regr']
        print('mean:', npz_dic['mean'].shape)
        print('std:', npz_dic['std'].shape)
        print('zs:', npz_dic['zs'])
        print('regr:', regr)
        # print('regr:', regr.predict([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))


if __name__ == "__main__":
    main()
