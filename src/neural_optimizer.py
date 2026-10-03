import tensorflow as tf
import numpy as np

class HybridSeriesLayer(tf.keras.layers.Layer):
    """
    Custom Keras Layer that locks in Symbolic Base Coefficients (C_base)
    and trains a Neural Refinement Delta (delta_C) so that:
    C_refined = C_base + delta_C
    """
    def __init__(self, c_base, **kwargs):
        super().__init__(**kwargs)
        self.c_base_init = np.array(c_base, dtype=np.float32)
        self.n_terms = len(self.c_base_init)

    def build(self, input_shape):
        # Anchor weights from SymPy / Symbolic module (Non-trainable reference)
        self.c_base = tf.constant(self.c_base_init, dtype=tf.float32, name="C_base")
        
        # Trainable neural refinement weights initialized to zero
        self.delta_c = self.add_weight(
            name="delta_C",
            shape=(self.n_terms,),
            initializer="zeros",
            trainable=True
        )

    def call(self, basis_matrix):
        c_refined = self.c_base + self.delta_c
        # Compute f_approx(x) = Phi(x) * C_refined
        return tf.linalg.matvec(basis_matrix, c_refined)

    def get_refined_coefficients(self):
        return (self.c_base + self.delta_c).numpy()


class NeuralCoefficientOptimizer:
    """
    Optimizes series coefficients using a custom hybrid loss function
    designed to suppress boundary divergence and Gibbs/Runge oscillations.
    """
    def __init__(self, c_base, learning_rate=0.01, alpha_peak=0.25, beta_tv=0.05, l2_reg=1e-4):
        self.c_base = c_base
        self.lr = learning_rate
        self.alpha_peak = alpha_peak  # Penalty weight for maximum overshoot (Runge/Gibbs)
        self.beta_tv = beta_tv        # Penalty weight for rapid oscillation (Total Variation)
        self.l2_reg = l2_reg          # Anchor regularization to keep delta_C bounded

        self.layer = HybridSeriesLayer(c_base)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_history = []

    def custom_hybrid_loss(self, y_true, y_pred):
        # 1. Standard Mean Squared Error (MSE)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Peak Overshoot Penalty (L-infinity norm approximation)
        max_error = tf.reduce_max(tf.abs(y_true - y_pred))

        # 3. Oscillation / Total Variation Penalty on prediction derivatives
        dy_pred = y_pred[1:] - y_pred[:-1]
         dy_true = y_true[1:] - y_true[:-1]
        oscillation_penalty = tf.reduce_mean(tf.abs(dy_pred - dy_true))

        # 4. Symbolic Anchor Regularization (penalize excessive drift from C_base)
        reg_penalty = tf.reduce_sum(tf.square(self.layer.delta_c))

        total_loss = (
            mse_loss 
            + (self.alpha_peak * max_error) 
            + (self.beta_tv * oscillation_penalty) 
            + (self.l2_reg * reg_penalty)
        )
        return total_loss

    def fit(self, basis_matrix, y_true, epochs=600, verbose=True):
        phi_tf = tf.constant(basis_matrix, dtype=tf.float32)
        y_tf = tf.constant(y_true, dtype=tf.float32)

        # Build layer
        _ = self.layer(phi_tf)

        for epoch in range(1, epochs + 1):
            with tf.GradientTape() as tape:
                y_pred = self.layer(phi_tf)
                loss = self.custom_hybrid_loss(y_tf, y_pred)

            gradients = tape.gradient(loss, self.layer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.layer.trainable_variables))
            self.loss_history.append(float(loss.numpy()))

            if verbose and epoch % 150 == 0:
                mse = float(tf.reduce_mean(tf.square(y_tf - y_pred)).numpy())
                print(f"Epoch {epoch:03d}/{epochs:03d} | Hybrid Loss: {loss.numpy():.6f} | Pure MSE: {mse:.6f}")

        return self.layer.get_refined_coefficients(), self.layer(phi_tf).numpy()