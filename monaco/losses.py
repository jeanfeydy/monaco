from pykeops.torch  import LazyTensor


def ED(x, y):
    """Computes the energy distance between two samples."""
    n, m = len(x), len(y) 

    x_i, x_j = LazyTensor(x[:,None,:]), LazyTensor(x[None,:,:]) # (n,1,D), (1,n,D)
    y_i, y_j = LazyTensor(y[:,None,:]), LazyTensor(y[None,:,:]) # (m,1,D), (1,m,D)

    D_xx = ((x_i - x_j)**2).sum(dim=2).sqrt().sum(dim=1).sum()
    D_xy = ((x_i - y_j)**2).sum(dim=2).sqrt().sum(dim=1).sum()
    D_yy = ((y_i - y_j)**2).sum(dim=2).sqrt().sum(dim=1).sum()

    return D_xy / (n*m) - .5 * (D_xx / (n*n) + D_yy / (m*m))


def Exponential(x, y, s = .1):
    """Computes an exponential kernel norm between two samples."""
    n, m = len(x), len(y) 

    x_i, x_j = LazyTensor(x[:,None,:]), LazyTensor(x[None,:,:]) # (n,1,D), (1,n,D)
    y_i, y_j = LazyTensor(y[:,None,:]), LazyTensor(y[None,:,:]) # (m,1,D), (1,m,D)

    D_xx = (-((x_i - x_j)**2).sum(dim=2).sqrt() / s).exp().sum(dim=1).sum()
    D_xy = (-((x_i - y_j)**2).sum(dim=2).sqrt() / s).exp().sum(dim=1).sum()
    D_yy = (-((y_i - y_j)**2).sum(dim=2).sqrt() / s).exp().sum(dim=1).sum()

    return D_xy / (n*m) - .5 * (D_xx / (n*n) + D_yy / (m*m))

