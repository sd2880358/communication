# 12/6
# LAMBDA = 50; EPOCHS=500;


## method 1 (without constrain noise):

- total_s_loss = identity_s + total_gen_loss + identity_g_loss
- total_n_loss = total_gen_loss + identity_g_loss
- total_i_loss = identity_g_loss + total_gen_loss
### Result in 500 epochs:
- total_loss = 0.0047
- signal_loss = 0.79
- noise_loss = 0.0856

## method 2 (constrain noise):

- total_s_loss = identity_s + total_gen_loss
- total_n_loss = identity_n + total_gen_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.011
- signal_loss = 0.766
- noise_loss = 0.058 


## method 3 (constrain noise):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.009
- signal_loss = 0.768
- noise_loss = 0.057

## method 4 (i = (g_i(g_s(i))))

- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.05
- signal_loss = 0.782
- noise_loss = 0.057

## method 5 (lambda = 100):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.011
- signal_loss = 0.761
- noise_loss = 0.058

## method 6 (constrain noise, lambda = 150):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.009
- signal_loss = 0.767
- noise_loss = 0.057

## method 7 (constrain noise, lambda = 100, epochs=100):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.056
- signal_loss = 0.723
- noise_loss = 0.058

## method 8 (constrain noise, lambda = 50):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.08
- signal_loss = 0.69
- noise_loss = 0.057
using cnn model, and acc = 0.20;

