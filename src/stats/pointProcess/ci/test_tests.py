
import ../
def test_timeRescaling():
    dt = 1e-3
    t0 = 0
    tf = 1
    cif = lambda(t): sin(2*pi*t)
    t = torch.linspace(t0, tf, tf/dt)


