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
- fake_total - features = 0.22

### Result in 500 epochs:

- total_loss = 0.011
- signal_loss = 0.761
- noise_loss = 0.058

CNN_model = 0.66

## method 6 (constrain noise, lambda = 150):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.009
- signal_loss = 0.767
- noise_loss = 0.057

## method 7 (constrain noise, lambda = 150, epochs=100):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.03
- signal_loss = 0.74
- noise_loss = 0.057

## method 8 (noise, lambda = 100, epochs=40):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.03
- signal_loss = 0.70
- noise_loss = 0.058
- fake_total - features = 0.26
using cnn model, and acc = 0.86;

## method 9 (lambda = 100, epochs = 160)

- total_s_loss = identity_s_loss + total_gen_loss + 0.5 * identity_g_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss + 0.5 * identity_g_loss
- total_i_loss = identity_g_loss + total_gen_loss

- total_loss = 0.03
- signal_loss = 0.16
- noise_loss = 0.058
- fake_total - features = 0.26