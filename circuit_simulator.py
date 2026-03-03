import numpy as np

class CircuitSimulator:
    def __init__(self, amplitude, frequency, R_test, C_test):
        """
        Class constructor.

        Args:
            amplitude:   Input signal amplitude.
            frequency:   Input signal frequency.
            R_test:      Resistor value.
            C_test:      Capacitor value.
        """
        # Initialize the MNA parameters
        self.amplitude = amplitude
        self.f = frequency
        self.R_test = R_test
        self.C_test = C_test
        self.G_mat = self.get_G(self.R_test)
        self.C_mat = self.get_C(self.C_test)
    def get_G(self, R):
        # Conductance matrix (G) representing static components (resistors/source connections)
        G_mat = np.array([
            [1/50, -1/50, 0, 1],
            [-1/50, 1/50, 0, 0],
            [0, 0, 1/R, 0],
            [1, 0, 0, 0]
        ])
        return G_mat

    def get_C(self, C):
        # Capacitance matrix (C) representing frequency-dependent components
        C_mat = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, C, 0],
            [0, 0, 0, 0]
        ])
        return C_mat

    def get_dGdR(self, R):
        # Derivative of the G matrix with respect to Resistance (used for sensitivity analysis)
        dGdR = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1/(R**2), 0],
            [0,0,0,0]
        ])
        return dGdR
    def get_dCdC(self):
        # Derivative of the C matrix with respect to Capacitance (used for sensitivity analysis)
        dCdC = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        return dCdC

    def get_sine(self, amplitude, frequency, t):
        # Generates the AC voltage input at time t
        return amplitude * np.sin(2 * np.pi * frequency * t)

    def get_b(self, amplitude, frequency, t):
        # Right-hand side vector (b) containing independent sources
        b = np.zeros((4,))
        b[3] = self.get_sine(amplitude, frequency, t)
        return b

    def get_f_vect(self, x):
        # Non-linear element vector for the diode of the half-wave rectifier
        # 1e-13 is saturation current; 0.025 is thermal voltage
        v1 = x[1]
        v2 = x[2]
        f_vect = np.zeros((4,))
        f_vect[1] = 1e-13 * (np.exp((v1 - v2) / 0.025) - 1)
        f_vect[2] = -1e-13 * (np.exp((v1 - v2) / 0.025) - 1)
        return f_vect

    def get_jac(self, x):
        # Jacobian of f(x) (diode injection vector)
        Is = 1e-13
        Vt = 0.025

        v1 = x[1]
        v2 = x[2]

        g = (Is / Vt) * np.exp((v1 - v2) / Vt)

        jac = np.zeros((4, 4))
        jac[1, 1] =  g
        jac[1, 2] = -g
        jac[2, 1] = -g
        jac[2, 2] =  g
        return jac    

    def BEuler(self, x_0, delta_t, T, noise = False):
        # Get the G and C matrices
        G = self.G_mat
        C = self.C_mat
        # Time-stepping simulation using the Backward Euler method
        x = x_0
        y = []
        tpoints = []
        t = 0
        while( t < T ):
            # Formulate the system: (G + C/dt) * x_new = b + (C/dt) * x_old
            A = G + (1/delta_t)*C
            b = self.get_b(self.amplitude, self.f, t) + (1/delta_t)*C@x
            # Solve non-linear equation for the current time step
            x = self.NewtonRaphson(A,b,x, 1e-6)
            tpoints.append(t)
            t += delta_t
            y.append(x)
        y = np.array(y)
        if noise:
            sigma = np.std(y,axis = 0)*0.025
            noise = np.random.normal(0, sigma, y.shape)
            y = y + noise
        tpoints = np.array(tpoints)
        return y, tpoints

    def NewtonRaphson(self, A, b, x, epsilon, max_iter=50):
        """
        Solve: A x + f(x) = b  (nonlinear due to diode f(x))
        using Newton-Raphson.

        Residual: r(x) = A x + f(x) - b
        Jacobian: J(x) = A + df/dx
        """
        x = x.copy()

        for _ in range(max_iter):
            f = self.get_f_vect(x)
            r = A @ x + f - b  # residual

            if np.linalg.norm(r, ord=2) < epsilon:
                return x

            J = A + self.get_jac(x)

            dx = np.linalg.solve(J, -r)
            x = x + dx

            if np.linalg.norm(dx, ord=2) < epsilon:
                return x

        return x

    def getSensitivities(self, x_pred, G_mat, C_mat, R, delta_t):
        # Calculates sensitivity of nodal voltages (x) to parameters R and C
        dxdr = []
        dxdc = []

        dGdR = self.get_dGdR(R)
        dCdC = self.get_dCdC()  # derivative of C_mat wrt scalar C

        for i in range(len(x_pred)):
            jac = self.get_jac(x_pred[i])

            A = G_mat + C_mat / delta_t + jac

            # ---- dx/dR ----
            if i == 0:
                tempr_r = -(dGdR @ x_pred[i])
            else:
                tempr_r = -(dGdR @ x_pred[i]) + (C_mat / delta_t) @ dxdr[i - 1]
            dxdr.append(np.linalg.solve(A, tempr_r))

            # ---- dx/dC ----
            if i == 0:
                tempr_c = -(dCdC / delta_t) @ x_pred[i]
            else:
                tempr_c = (
                    -(dCdC / delta_t) @ x_pred[i]
                    + (dCdC / delta_t) @ x_pred[i - 1]
                    + (C_mat / delta_t) @ dxdc[i - 1]
                )
            dxdc.append(np.linalg.solve(A, tempr_c))

        dxdr = np.array(dxdr)
        dxdc = np.array(dxdc)
        return dxdr, dxdc

    def GaussNewton(self, R_init, C_init, x_init, x_test, delta_t, T, max_iter = 100, noise = False):
        # Parameter estimation: finds R and C that best fit the observed 'x_test' data
        R_pred = R_init
        C_pred = C_init
        cost = 1
        iter = 0
        x_init = x_init.squeeze()

        while (iter < max_iter):
            self.G_mat = self.get_G(R_pred)
            self.C_mat = self.get_C(C_pred)
            # 1. Run simulation with current parameter guesses
            x_pred, _ = self.BEuler(x_init, delta_t, T, noise = noise)
            # 2. Calculate residuals (error)
            r = (x_test - x_pred).reshape(-1,1)
            cost = 0.5*np.sum(r**2)

            # 3. Calculate Jacobian of the error with respect to R and C (Sensitivities)
            dxdr, dxdc = self.getSensitivities(x_pred, self.G_mat, self.C_mat, R_pred, delta_t)
            S_r = (dxdr * R_pred).reshape(-1,1)
            S_c = (dxdc * C_pred).reshape(-1,1)
            J_r = np.hstack([S_r, S_c])

            # 4. Gauss-Newton update step
            beta = (np.linalg.inv(J_r.T@J_r))@(J_r.T@r).squeeze()
            R_pred = R_pred * np.exp(beta[0])
            C_pred = C_pred * np.exp(beta[1])
            iter += 1

        print(f"Number of iterations: {iter}.")
        return R_pred, C_pred, cost
