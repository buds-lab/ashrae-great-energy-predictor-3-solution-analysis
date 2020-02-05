import numpy as np


class FastLabelEncoder():
    """Map categorical variable into {0, 1, ..., n_categories}. 
    
    Note: https://stackoverflow.com/questions/45321999/how-can-i-optimize-label-encoding-for-large-data-sets-sci-kit-learn?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa    
    """
    def __init__(self):
        self.lookup = None
    
    def fit(self, x): 
        labels = np.unique(x, return_inverse=True)[1]
        self.lookup = dict(zip(x.flatten(),labels))    
        
    def transform(self, x): 
        return np.vectorize(self.lookup.get)(x)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)    


class GaussianTargetEncoder():
        
    def __init__(self, group_cols, target_col="target", prior_cols=None):
        self.group_cols = group_cols
        self.target_col = target_col
        self.prior_cols = prior_cols

    def _get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(1)
        return prior
                    
    def fit(self, df):
        self.stats = df.assign(mu_prior=self._get_prior(df), y=df[self.target_col])
        self.stats = self.stats.groupby(self.group_cols).agg(
            n        = ("y", "count"),
            mu_mle   = ("y", np.mean),
            sig2_mle = ("y", np.var),
            mu_prior = ("mu_prior", np.mean),
        )        
    
    def transform(self, df, prior_precision=1e-6, stat_type="mean"):
        
        precision = prior_precision + self.stats.n/self.stats.sig2_mle
        
        if stat_type == "mean":
            numer = prior_precision*self.stats.mu_prior\
                    + self.stats.n/self.stats.sig2_mle*self.stats.mu_mle
            denom = precision
        elif stat_type == "var":
            numer = 1.0
            denom = precision
        elif stat_type == "precision":
            numer = precision
            denom = 1.0
        else: 
            raise ValueError(f"stat_type={stat_type} not recognized.")
        
        mapper = dict(zip(self.stats.index, numer / denom))
        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])
        
        values = np.array([mapper.get(k) for k in keys]).astype(float)
        
        prior = self._get_prior(df)
        values[~np.isfinite(values)] = prior[~np.isfinite(values)]
        
        return values
    
    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)
