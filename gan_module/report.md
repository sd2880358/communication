# 12/6
# LAMBDA = 50; EPOCHS=500;


## method 1 (without constrain noise):

- total_s_loss = identity_s + total_gen_loss + identity_g_loss
- total_n_loss = total_gen_loss + identity_g_loss
- total_i_loss = identity_g_loss + total_gen_loss
### Result in 500 epochs:
- total_loss = 0.015
- signal_loss = 0.769
- noise_loss = 0.153

## method 2 (constrain noise):

- total_s_loss = identity_s + total_gen_loss
- total_n_loss = identity_n + total_gen_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.011
- signal_loss = 0.768
- noise_loss = 0.058 


## method 3 (without constrain noise):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.009
- signal_loss = 0.767
- noise_loss = 0.057

## method 4 (i = (g_i(g_s(i))))

- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.05
- signal_loss = 0.782
- noise_loss = 0.057

## method 3 (without constrain noise, lambda = 100):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_n_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.009
- signal_loss = 0.767
- noise_loss = 0.057

