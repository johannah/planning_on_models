# modifying and extending code from kkastner to pytorch -
# https://gist.github.com/kastnerkyle/d1f027423ba9c363236bd9fcb799cc8d
# kkasters implementation based off of below
# https://github.com/skerit/cmusphinx/blob/master/SphinxTrain/python/cmusphinx/divergence.py#L47
import numpy as np
import itertools
from IPython import embed

def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1. / qv
    # Difference between means pm, qm
    diff = qm - pm
    p1 = np.log(dqv / dpv)# log |\Sigma_q| / |\Sigma_p|
    p2 = (iqv * pv).sum(axis)# + tr(\Sigma_q^{-1} * \Sigma_p)
    p3 = (diff * iqv * diff).sum(axis)# + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
    p4 = len(pm)# - N
    return 0.5 * (p1 + p2 + p3 - p4)


def gau_kl2(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussians pm,pv to Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    returns KL of each G in pm, pv to all qm, qv
    """
    axis1 = 1
    axis2 = 2
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod(axis1)
    dqv = qv.prod(axis1)
    # Inverse of diagonal covariance qv
    iqv = 1. / qv
    # Difference between means pm, qm
    diff = qm - pm[:, None]
    p1 = np.log(dqv / dpv[:, None])
    p2 = (iqv * pv[:, None]).sum(axis2)
    p3 = (diff * iqv[None] * diff).sum(axis2)
    p4 = pm.shape[1]
    return 0.5 * (p1 + p2 + p3 - p4)

def create_mixture(n_dim, n_mixtures):
    this_mixture = []
    ws = []
    for k in range(n_mixtures):
        # get weights first, will sum to 1
        ws.append(random_state.rand())
    ws = np.array(ws) / np.sum(ws)

    for k in range(n_mixtures):
        # draw means from range -5, 5
        # draw variances from 0.01, 20
        mu = random_state.rand(n_dim) * 10 - 5
        var = random_state.rand(n_dim) * 19.99 + 0.01
        # will be a triple of weight, means, variances
        this_mixture.append((ws[k], mu, var))
    return this_mixture



# https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf
n_dim = 20
n_mixtures = 8
n_comparisons = 3
random_state = np.random.RandomState(2188)
mixtures = []
for i in range(n_comparisons):
    mixtures.append((i, create_mixture(n_dim, n_mixtures)))

def extract(mix):
    _pis = [m[0] for m in mix]
    _mus = [m[1] for m in mix]
    _vars = [m[2] for m in mix]
    return np.array(_pis), np.array(_mus), np.array(_vars)

# compare kld between all pairs
klds = {}

for cnt, (a, b) in enumerate(itertools.product(mixtures, repeat=2)):
    print(cnt)
    idx1 = a[0]
    idx2 = b[0]
    a_pis, a_mus, a_vars = extract(a[1])
    b_pis, b_mus, b_vars = extract(b[1])
    kl = 0.
    # KL approx by this value, eq. (18)
    # LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
    #DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS
    # https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf
    """
    # slow, loopy version
    for i in range(len(a_mus)):
        num = 0.
        for j in range(len(a_mus)):
            num += a_pis[j] * np.exp(-gau_kl(a_mus[i], a_vars[i], a_mus[j], a_vars[j]))
        den = 0.
        for j in range(len(b_mus)):
            den += b_pis[j] * np.exp(-gau_kl(a_mus[i], a_vars[i], b_mus[j], b_vars[j]))
        kl_tmp = a_pis[i] * (np.log(num / den))
        kl += kl_tmp
    """

    """
    # faster, less loopy version
    for i in range(len(a_mus)):
        num = np.sum(a_pis * np.exp(-gau_kl(a_mus[i], a_vars[i], a_mus, a_vars)))
        den = np.sum(b_pis * np.exp(-gau_kl(a_mus[i], a_vars[i], b_mus, b_vars)))
        kl_tmp = a_pis[i] * (np.log(num / den))
        kl += kl_tmp
    """

    # vectorized
    nums = np.sum(a_pis * np.exp(-gau_kl2(a_mus, a_vars, a_mus, a_vars)), axis=1)
    dens = np.sum(b_pis * np.exp(-gau_kl2(a_mus, a_vars, b_mus, b_vars)), axis=1)
    kl = np.sum(a_pis * np.log(nums / dens))
    key = "{}||{}".format(idx1, idx2)
    klds[key] = kl
    print(key, kl)
embed()
